from __future__ import annotations

import json
import os
import shutil
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

# Return a compact UTC timestamp string
def now_iso():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

# Lightweight persistent memory for agent runs
# Stored sections:
#  - dataset_records: keyed by exact dataset fingerprint
#   - run_history: compact recent run trail
#   - notes: optional debug / failure notes
class JSONMemory:
    def __init__(self, path = "agent_memory.json"):
        self.path = path
        self.data: Dict[str, Any] = {
            "dataset_records": {},
            "run_history": [],
            "notes": [],
            "version": 2,
        }
        self.load()

    # Load memory from disk, with a safe fallback if the file is corrupted
    def load(self):
        if not os.path.exists(self.path):
            return

        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                loaded = json.load(handle)

            if "datasets" in loaded and "dataset_records" not in loaded:
                self.data["dataset_records"] = loaded.get("datasets", {})
                self.data["notes"] = loaded.get("notes", [])
            else:
                self.data.update(loaded)
        except Exception:
            backup = self.path + ".bak"
            shutil.copy(self.path, backup)
            self.data = {
                "dataset_records": {},
                "run_history": [],
                "notes": [{"ts": now_iso(), "msg": f"Memory was reset; backup saved to {backup}"}],
                "version": 2,
            }

    # Persist current memory to disk
    def save(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump(self.data, handle, indent=2, ensure_ascii=False)

    # Append a note and save immediately
    def add_note(self, msg):
        self.data.setdefault("notes", []).append({"ts": now_iso(), "msg": msg})
        self.save()

    # Return a deep copy of an exact dataset record, if available
    def get_dataset_record(self, fingerprint):
        return deepcopy(self.data.get("dataset_records", {}).get(fingerprint))

    # Extract only the high level profile properties used for approximate matching
    def _profile_signature(self, profile):
        return {
            "size_bucket": profile.get("size_bucket"),
            "rows": int(profile.get("shape", {}).get("rows", 0)),
            "cols": int(profile.get("shape", {}).get("cols", 0)),
            "numeric_ratio": float(profile.get("feature_balance", {}).get("numeric_ratio", 0.0)),
            "categorical_ratio": float(profile.get("feature_balance", {}).get("categorical_ratio", 0.0)),
            "missing_severity": profile.get("missingness_summary", {}).get("severity", "low"),
            "overall_missing": float(profile.get("missingness_summary", {}).get("overall_pct", 0.0)),
            "imbalance_ratio": float(profile.get("imbalance_ratio") or 1.0),
            "num_classes": int(profile.get("num_classes", 0) or 0),
            "high_dimensional": bool(profile.get("high_dimensional", False)),
            "high_cardinality_count": len(profile.get("high_cardinality_features", [])),
            "dominant_feature_type": profile.get("dominant_feature_type", "mixed"),
            "weak_signal_risk": bool(profile.get("weak_signal_risk", False)),
        }

    # Ensure all expected memory counters exist
    def normalise(self, record):
        copied = deepcopy(record)
        copied.setdefault("runs", [])
        copied.setdefault("successful_models", {})
        copied.setdefault("failed_models", {})
        copied.setdefault("successful_strategies", {})
        copied.setdefault("unsuccessful_strategies", {})
        copied.setdefault("root_cause_counts", {})
        copied.setdefault("recent_summary", {})
        return copied

    # Insert or update a dataset record and roll up useful counts from the latest run
    def upsert_dataset_record(self, fingerprint, record):
        existing = self.normalise(self.data.setdefault("dataset_records", {}).get(fingerprint, {}))
        incoming = deepcopy(record)

        if "profile_signature" not in incoming and "profile" in incoming:
            incoming["profile_signature"] = self._profile_signature(incoming["profile"])

        latest_run = incoming.get("latest_run")
        if latest_run is not None:
            existing["runs"].append(latest_run)
            existing["runs"] = existing["runs"][-10:]

            best_model = latest_run.get("best_model")
            best_score = float(latest_run.get("best_score", 0.0) or 0.0)
            root_causes = latest_run.get("root_causes", []) or []

            if best_model:
                bucket = existing["successful_models"] if best_score >= 0.70 else existing["failed_models"]
                bucket[best_model] = bucket.get(best_model, 0) + 1

            for flag, enabled in latest_run.get("strategy_flags", {}).items():
                if not enabled:
                    continue
                strategy_bucket = (
                    existing["successful_strategies"] if best_score >= 0.70 else existing["unsuccessful_strategies"]
                )
                strategy_bucket[flag] = strategy_bucket.get(flag, 0) + 1

            for cause in root_causes:
                existing["root_cause_counts"][cause] = existing["root_cause_counts"].get(cause, 0) + 1

            self.data.setdefault("run_history", []).append(
                {
                    "ts": latest_run.get("ts", now_iso()),
                    "fingerprint": fingerprint,
                    "best_model": best_model,
                    "best_score": best_score,
                }
            )
            self.data["run_history"] = self.data["run_history"][-50:]

        for key, value in incoming.items():
            if key == "latest_run":
                continue
            existing[key] = value

        existing["updated_at"] = now_iso()
        self.data.setdefault("dataset_records", {})[fingerprint] = existing
        self.save()

    # Compute a simple weighted similarity score between two dataset signatures
    def similarity(self, a, b):
        score = 0.0
        weight = 0.0

        def ratio_similarity(x, y, cap):
            if cap <= 0:
                return 1.0
            return max(0.0, 1.0 - min(abs(x - y) / cap, 1.0))

        score += 2.0 * ratio_similarity(float(a.get("numeric_ratio", 0.0)), float(b.get("numeric_ratio", 0.0)), 1.0)
        weight += 2.0

        score += 1.5 if a.get("size_bucket") == b.get("size_bucket") else 0.0
        weight += 1.5

        score += 1.5 if a.get("missing_severity") == b.get("missing_severity") else 0.0
        weight += 1.5

        score += 1.5 if a.get("dominant_feature_type") == b.get("dominant_feature_type") else 0.0
        weight += 1.5

        score += 1.5 * ratio_similarity(float(a.get("imbalance_ratio", 1.0)), float(b.get("imbalance_ratio", 1.0)), 8.0)
        weight += 1.5

        score += 1.0 if int(a.get("num_classes", 0)) == int(b.get("num_classes", 0)) else 0.0
        weight += 1.0

        score += 1.0 if bool(a.get("high_dimensional", False)) == bool(b.get("high_dimensional", False)) else 0.0
        weight += 1.0

        score += 1.0 * ratio_similarity(
            float(a.get("high_cardinality_count", 0)),
            float(b.get("high_cardinality_count", 0)),
            10.0,
        )
        weight += 1.0

        score += 1.0 if bool(a.get("weak_signal_risk", False)) == bool(b.get("weak_signal_risk", False)) else 0.0
        weight += 1.0

        return score / weight if weight else 0.0

    # Retrieve soft guidance from similar past datasets
    def get_similar_dataset_hint(self, profile, min_similarity = 0.45, top_k = 3):
        target_signature = self._profile_signature(profile)
        scored: List[Tuple[float, str, Dict[str, Any]]] = []

        for fingerprint, record in self.data.get("dataset_records", {}).items():
            normalised = self.normalise(record)
            signature = normalised.get("profile_signature")
            if not signature:
                continue

            similarity = self.similarity(target_signature, signature)
            if similarity >= min_similarity:
                scored.append((similarity, fingerprint, normalised))

        if not scored:
            return None

        scored.sort(key=lambda item: item[0], reverse=True)
        top = scored[:top_k]

        preferred_models: Dict[str, float] = {}
        avoid_models: Dict[str, float] = {}
        strategy_votes: Dict[str, float] = {}
        root_cause_votes: Dict[str, float] = {}
        matched_runs = 0

        for similarity, _, record in top:
            for model_name, count in record.get("successful_models", {}).items():
                preferred_models[model_name] = preferred_models.get(model_name, 0.0) + similarity * count

            for model_name, count in record.get("failed_models", {}).items():
                avoid_models[model_name] = avoid_models.get(model_name, 0.0) + similarity * count

            for flag_name, count in record.get("successful_strategies", {}).items():
                strategy_votes[flag_name] = strategy_votes.get(flag_name, 0.0) + similarity * count

            for cause_name, count in record.get("root_cause_counts", {}).items():
                root_cause_votes[cause_name] = root_cause_votes.get(cause_name, 0.0) + similarity * count

            matched_runs += len(record.get("runs", []))

        preferred_sorted = sorted(preferred_models.items(), key=lambda item: item[1], reverse=True)
        avoid_sorted = sorted(avoid_models.items(), key=lambda item: item[1], reverse=True)
        strategy_sorted = sorted(strategy_votes.items(), key=lambda item: item[1], reverse=True)
        root_cause_sorted = sorted(root_cause_votes.items(), key=lambda item: item[1], reverse=True)

        best_record = top[0][2]
        best_run = best_record.get("runs", [])[-1] if best_record.get("runs") else {}

        return {
            "source": "similar_dataset",
            "similarity": round(top[0][0], 3),
            "matched_runs": matched_runs,
            "best_model": best_run.get("best_model") or best_record.get("best_model"),
            "preferred_models": preferred_sorted[:5],
            "avoid_models": [name for name, _ in avoid_sorted[:3]],
            "strategy_preferences": strategy_sorted[:5],
            "common_root_causes": [name for name, _ in root_cause_sorted[:3]],
            "recent_summary": best_record.get("recent_summary", {}),
        }