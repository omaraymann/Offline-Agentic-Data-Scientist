from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
from sklearn import preprocessing

from agents.memory import JSONMemory
from agents.planner import create_plan
from agents.reflector import apply_replan_strategy, reflect, should_replan
from tools.data_profiler import dataset_fingerprint, infer_target_column, profile_dataset
from tools.evaluation import evaluate_best, save_json, write_markdown_report
from tools.modelling import build_preprocessor, select_models, train_models

# Run specific metadata saved and reused across the pipeline
@dataclass
class RunContext:

    run_id: str
    started_at: str
    data_path: str
    target: str
    output_dir: str
    seed: int
    test_size: float
    max_replans: int

# Return a compact UTC timestamp string
def now_iso():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


class AgenticDataScientist:
    """
    Offline agentic pipeline for tabular classification tasks.

    The agent follows this high-level flow:
    1. Load dataset
    2. Resolve target
    3. Profile dataset
    4. Retrieve memory hints
    5. Create plan
    6. Train and evaluate candidate models
    7. Reflect on results
    8. Replan if needed
    9. Persist outputs and update memory
    """

    def __init__(self, memory_path = "agent_memory.json", verbose = True):
        self.verbose = verbose
        self.memory = JSONMemory(memory_path)
        self.ctx: Optional[RunContext] = None
        self.state: Dict[str, Any] = {}

    # Print log messages when verbose mode is enabled
    def log(self, message):
        if self.verbose:
            print(f"[AgenticDataScientist] {message}")

    # Load a CSV dataset from disk
    def load_data(self, path) :
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset not found: {path}")

        self.log(f"Loading dataset: {path}")
        df = pd.read_csv(path)
        self.log(f"Loaded {df.shape[0]} rows × {df.shape[1]} cols")
        return df

    # "Create output directory and initialise run state
    def initialise_context(self, data_path, target, output_root, seed, test_size, max_replans):
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8]
        output_dir = os.path.join(output_root, run_id)
        os.makedirs(output_dir, exist_ok=True)

        self.ctx = RunContext(
            run_id=run_id,
            started_at=now_iso(),
            data_path=data_path,
            target=target,
            output_dir=output_dir,
            seed=seed,
            test_size=test_size,
            max_replans=max_replans,
        )
        self.state = {
            "replan_count": 0,
            "attempt_history": [],
            "execution_log": [],
            "memory_hint": None,
            "fingerprint": None,
            "selection_rationale": None,
        }

    # Append a structured log event to execution history
    def record_log(self, stage, message):
        event = {"ts": now_iso(), "stage": stage, "message": message}
        self.state.setdefault("execution_log", []).append(event)
        self.log(f"{stage}: {message}")

    # Resolve the target column
    # If the user passes 'auto', the system will infer the most likely target
    def resolve_target(self, df, target):
        if target.strip().lower() != "auto":
            return {
                "target": target,
                "target_inference": {
                    "method": "manual",
                    "confidence": "high",
                    "candidates": [target],
                },
            }

        inferred, metadata = infer_target_column(df, return_metadata=True)
        if not inferred:
            raise ValueError("Could not infer target column. Please provide --target <name>.")

        self.record_log(
            "target",
            f"Inferred target '{inferred}' using method={metadata.get('method')}",
        )
        return {"target": inferred, "target_inference": metadata}

    # Convert a stored dataset record into planner-friendly memory guidance
    def build_memory_hint_from_record(self, exact_record):
        return {
            "source": "exact_fingerprint",
            "best_model": exact_record.get("best_model"),
            "preferred_models": sorted(
                exact_record.get("successful_models", {}).items(),
                key=lambda item: item[1],
                reverse=True,
            )[:5],
            "avoid_models": [
                name
                for name, _ in sorted(
                    exact_record.get("failed_models", {}).items(),
                    key=lambda item: item[1],
                    reverse=True,
                )[:3]
            ],
            "strategy_preferences": sorted(
                exact_record.get("successful_strategies", {}).items(),
                key=lambda item: item[1],
                reverse=True,
            )[:5],
            "recent_summary": exact_record.get("recent_summary"),
        }

    # Create the dataset profile and enrich it with memory guidance when available
    def prepare_profile(self, df, target, target_inference):
        profile = profile_dataset(df, target=target, target_inference=target_inference)
        fingerprint = dataset_fingerprint(df, target)
        self.state["fingerprint"] = fingerprint

        exact_record = self.memory.get_dataset_record(fingerprint)
        if exact_record:
            memory_hint = self.build_memory_hint_from_record(exact_record)
            self.record_log("memory", f"Exact memory hit for {fingerprint}")
        else:
            memory_hint = self.memory.get_similar_dataset_hint(profile)
            if memory_hint:
                self.record_log(
                    "memory",
                    f"Approximate memory hit with similarity={memory_hint.get('similarity')}",
                )
            else:
                self.record_log("memory", "No relevant prior memory found")

        self.state["memory_hint"] = memory_hint

        if memory_hint:
            profile["memory_guidance"] = memory_hint
            for flag_name, _ in memory_hint.get("strategy_preferences", []):
                if flag_name in profile.get("strategy_flags", {}):
                    profile["strategy_flags"][flag_name] = True

        return profile

    # Summarise why the final model was selected.
    # This makes the run easier to justify in the report and demo.
    def build_selection_rationale(self, evaluation, profile):
        all_metrics = evaluation.get("all_metrics", [])
        best = evaluation.get("best_metrics", {})
        primary_metric = evaluation.get("primary_metric", "accuracy")
        compare_on_cv = bool(profile.get("strategy_flags", {}).get("compare_on_cv", False))

        runner_up = None
        ordered = sorted(
            all_metrics,
            key=lambda item: (
                float(item.get(primary_metric, 0.0)),
                float(item.get("f1_macro", 0.0)),
            ),
            reverse=True,
        )
        if len(ordered) > 1:
            runner_up = ordered[1]

        rationale = {
            "selected_model": best.get("model"),
            "primary_metric": primary_metric,
            "compare_on_cv": compare_on_cv,
            "selection_policy": (
                "cv_first_then_holdout"
                if compare_on_cv
                else "holdout_first_then_cv"
            ),
            "best_primary_metric": best.get(primary_metric),
            "best_cv_balanced_accuracy_mean": best.get("cv_balanced_accuracy_mean"),
            "best_cv_balanced_accuracy_std": best.get("cv_balanced_accuracy_std"),
            "runner_up_model": None if runner_up is None else runner_up.get("model"),
            "runner_up_primary_metric": None if runner_up is None else runner_up.get(primary_metric),
            "runner_up_cv_balanced_accuracy_mean": None if runner_up is None else runner_up.get("cv_balanced_accuracy_mean"),
        }

        if compare_on_cv:
            rationale["human_summary"] = (
                "Model selected using cross-validation-aware ranking to prefer stability "
                "over a single holdout split."
            )
        else:
            rationale["human_summary"] = (
                "Model selected mainly from holdout performance, with CV used as supporting evidence."
            )

        return rationale

    # Convert high level planner steps into concrete runtime flags
    # This makes the plan operational rather than only descriptive
    def apply_plan_runtime_overrides(self, profile, plan):
        updated = dict(profile)
        flags = dict(updated.get("strategy_flags", {}))

        step_set = set(plan)

        if "activate_imbalance_strategy" in step_set:
            flags["use_class_weight"] = True

        if "tune_decision_thresholds" in step_set:
            flags["tune_decision_threshold"] = True

        if "validate_with_cross_validation" in step_set:
            flags["enable_cv"] = True

        if "rank_with_cv_stability" in step_set:
            flags["compare_on_cv"] = True

        if "prefer_simple_models" in step_set or "limit_model_complexity" in step_set:
            flags["prefer_simple_models"] = True

        if "expand_model_pool" in step_set:
            flags["expand_model_diversity"] = True

        if "control_high_cardinality" in step_set:
            flags["group_rare_categories"] = True

        if "drop_high_missing_columns" in step_set:
            flags["drop_high_missing_features"] = True

        if "include_kernel_models" in step_set:
            flags["include_kernel_model"] = True

        if "exclude_boosting_models" in step_set:
            flags["include_boosting"] = False
            flags["include_gradient_boosting"] = False

        if "exclude_extra_trees_models" in step_set:
            flags["include_extra_trees"] = False

        updated["strategy_flags"] = flags
        return updated

    # Execute one complete attempt:
    # preprocessing -> model selection -> training -> evaluation -> reflection
    def run_single_attempt(self, df, profile, plan, attempt_index):
        assert self.ctx is not None

        self.record_log("attempt", f"Starting attempt {attempt_index}")

        runtime_profile = self.apply_plan_runtime_overrides(profile, plan)
        self.record_log(
            "plan_runtime",
            f"Applied runtime plan overrides using {len(plan)} planner steps",
        )

        preprocessor = build_preprocessor(runtime_profile)
        candidates = select_models(runtime_profile, seed=self.ctx.seed)
        candidate_names = [name for name, _ in candidates]
        self.record_log("models", f"Candidate models: {candidate_names}")

        results = train_models(
            df=df,
            target=self.ctx.target,
            preprocessor=preprocessor,
            candidates=candidates,
            profile=runtime_profile,
            seed=self.ctx.seed,
            test_size=self.ctx.test_size,
            output_dir=self.ctx.output_dir,
            verbose=self.verbose,
        )

        evaluation = evaluate_best(
            results,
            output_dir=self.ctx.output_dir,
            profile=runtime_profile,
        )
        reflection = reflect(
            dataset_profile=runtime_profile,
            evaluation=evaluation,
            all_metrics=evaluation["all_metrics"],
        )

        selection_rationale = self.build_selection_rationale(evaluation, runtime_profile)
        self.state["selection_rationale"] = selection_rationale

        attempt_summary = {
            "attempt_index": attempt_index,
            "best_model": evaluation["best_metrics"].get("model"),
            "balanced_accuracy": evaluation["best_metrics"].get("balanced_accuracy"),
            "f1_macro": evaluation["best_metrics"].get("f1_macro"),
            "reflection_status": reflection.get("status"),
        }
        self.state.setdefault("attempt_history", []).append(attempt_summary)

        save_json(os.path.join(self.ctx.output_dir, f"attempt_{attempt_index}_metrics.json"), evaluation)
        save_json(os.path.join(self.ctx.output_dir, f"attempt_{attempt_index}_reflection.json"), reflection)
        save_json(
            os.path.join(self.ctx.output_dir, f"attempt_{attempt_index}_selection_rationale.json"),
            selection_rationale,
        )

        return {
            "evaluation": evaluation,
            "reflection": reflection,
            "results": results,
            "plan": plan,
            "profile": runtime_profile,
            "selection_rationale": selection_rationale,
        }

    # Write all final run artefacts to disk
    def persist_final_outputs( self, profile, plan, evaluation, reflection, attempt_index):
        assert self.ctx is not None

        save_json(os.path.join(self.ctx.output_dir, "eda_summary.json"), profile)
        save_json(
            os.path.join(self.ctx.output_dir, "plan.json"),
            {"plan": plan, "attempt_count": attempt_index + 1},
        )
        save_json(os.path.join(self.ctx.output_dir, "metrics.json"), evaluation)
        save_json(os.path.join(self.ctx.output_dir, "reflection.json"), reflection)
        save_json(os.path.join(self.ctx.output_dir, "execution_log.json"), self.state.get("execution_log", []))
        save_json(
            os.path.join(self.ctx.output_dir, "selection_rationale.json"),
            self.state.get("selection_rationale"),
        )
        save_json(
            os.path.join(self.ctx.output_dir, "attempt_history.json"),
            self.state.get("attempt_history", []),
        )

        write_markdown_report(
            out_path=os.path.join(self.ctx.output_dir, "report.md"),
            ctx=self.ctx,
            fingerprint=self.state["fingerprint"],
            dataset_profile=profile,
            plan=plan,
            eval_payload=evaluation,
            reflection=reflection,
            memory_hint=self.state.get("memory_hint"),
            attempt_index=attempt_index,
            selection_rationale=self.state.get("selection_rationale"),
        )

    # Store useful run outcomes for future exact or approximate reuse
    def update_memory(self, profile, evaluation, reflection):
        assert self.ctx is not None

        best_metrics = evaluation["best_metrics"]
        best_score = float(best_metrics.get("balanced_accuracy", 0.0))
        fingerprint = self.state["fingerprint"]

        self.memory.upsert_dataset_record(
            fingerprint,
            {
                "target": self.ctx.target,
                "shape": profile["shape"],
                "profile_signature": self.memory._profile_signature(profile),
                "best_model": best_metrics.get("model"),
                "best_metrics": best_metrics,
                "recent_summary": {
                    "selection_rationale": self.state.get("selection_rationale"),
                    "root_causes": reflection.get("root_causes", []),
                    "status": reflection.get("status"),
                },
                "latest_run": {
                    "ts": now_iso(),
                    "run_id": self.ctx.run_id,
                    "best_model": best_metrics.get("model"),
                    "best_score": best_score,
                    "strategy_flags": profile.get("strategy_flags", {}),
                    "root_causes": reflection.get("root_causes", []),
                },
            },
        )

    def run(self, data_path, target, output_root = "outputs", seed = 42, test_size = 0.2, max_replans = 1,):
        self.initialise_context(
            data_path=data_path,
            target=target,
            output_root=output_root,
            seed=seed,
            test_size=test_size,
            max_replans=max_replans,
        )
        assert self.ctx is not None

        try:
            df = self.load_data(data_path)
            resolved_target = self.resolve_target(df, target)
            self.ctx.target = resolved_target["target"]

            profile = self.prepare_profile(
                df=df,
                target=self.ctx.target,
                target_inference=resolved_target["target_inference"],
            )
            plan = create_plan(profile, memory_hint=self.state.get("memory_hint"))
            self.record_log("plan", f"Initial plan has {len(plan)} steps")

            final_payload: Optional[Dict[str, Any]] = None
            attempt_index = 0

            while True:
                final_payload = self.run_single_attempt(
                    df=df,
                    profile=profile,
                    plan=plan,
                    attempt_index=attempt_index,
                )
                evaluation = final_payload["evaluation"]
                reflection = final_payload["reflection"]

                should_try_again = should_replan(
                    reflection,
                    current_replans=self.state["replan_count"],
                    max_replans=self.ctx.max_replans,
                    history=self.state.get("attempt_history", []),
                )

                if not should_try_again:
                    self.record_log("replan", "No further replanning required")
                    break

                self.state["replan_count"] += 1
                attempt_index += 1
                self.record_log("replan", f"Applying replan strategy #{self.state['replan_count']}")
                plan, profile = apply_replan_strategy(plan, profile, reflection)

            assert final_payload is not None
            self.persist_final_outputs(
                profile=final_payload["profile"],
                plan=final_payload["plan"],
                evaluation=final_payload["evaluation"],
                reflection=final_payload["reflection"],
                attempt_index=attempt_index,
            )
            self.update_memory(
                profile=final_payload["profile"],
                evaluation=final_payload["evaluation"],
                reflection=final_payload["reflection"],
            )

            self.record_log("done", f"Outputs saved to {self.ctx.output_dir}")
            return self.ctx.output_dir

        except Exception as exc:
            error_payload = {
                "status": "failed",
                "error": str(exc),
                "run_id": self.ctx.run_id if self.ctx else None,
            }

            if self.ctx is not None:
                save_json(os.path.join(self.ctx.output_dir, "error.json"), error_payload)
                save_json(
                    os.path.join(self.ctx.output_dir, "execution_log.json"),
                    self.state.get("execution_log", []),
                )

            self.memory.add_note(f"Run failed for {data_path}: {exc}")
            raise