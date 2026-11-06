from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
from pathlib import Path
import time


SETTINGS = {
    "current_25": (25., None),
    "baseline_50": (50., None),
    "current_100": (100., None),
    "current_200": (200., None),
    "relay_Y_1_18cm": (50., {"enabled": True, "count": 1, "spacing_cm": 18., "axis": "Y"}),
    "relay_Z_1_18cm": (50., {"enabled": True, "count": 1, "spacing_cm": 18., "axis": "Z"}),
    "relay_Z_4_18cm": (50., {"enabled": True, "count": 4, "spacing_cm": 18., "axis": "Z"}),
    "current_100_relay_Z": (100., {"enabled": True, "count": 1, "spacing_cm": 18., "axis": "Z"}),
}


def run_case(case: tuple) -> dict:
    import numpy as np

    from evaluation.cluster_scenarios_v2_1 import configure_cluster_scene
    from simulation.mi_settings_v2_1 import default_settings, design_values, make_settings
    from simulation.perception_v2_1 import RobotPerception
    from simulation.relay_settings_v2_1 import with_relay
    from web_runtime_v2_1 import WebSimulationSession

    setting, scenario, environment, seed, steps = case
    started = time.perf_counter()
    current, relay = SETTINGS[setting]
    base = default_settings()
    profile = make_settings(base, {**design_values(base), "tx_current_rms_a": current},
                            "예상값", "coil_input", [20, 30, 40, 50, 60, 80], [None] * 6)
    if relay:
        profile = with_relay(profile, relay)
    session = WebSimulationSession()
    session.reset(seed, environment, profile)
    board = session.board
    if scenario == "cluster_none":
        configure_cluster_scene(board, "none")
    elif scenario == "no_people":
        board.people = []
        board.measurements = [board.sensor.measure(board)]
        board.magnetic_scan = board.sensor.scan(board)
    board.perception = RobotPerception()
    board.perception.update(board.robot.position, board.magnetic_scan, board.range_scan)
    board.detection_tracker.clear()
    session.detector_runtime.reset()
    session.coverage_runtime.reset()
    session.rescue_manager.reset()
    session._update_detector()
    initial_people = len(board.people)
    events = []
    frames = predicted = in_range = 0
    path_length = 0.
    proposal_frames = []
    for index in range(steps):
        if session.rescue_manager.all_rescued(board):
            break
        rescued_before = set(session.rescue_manager.rescued_indices)
        previous = board.robot.position.copy()
        session.step()
        frames += 1
        path_length += float(np.linalg.norm(board.robot.position - previous))
        result = session.detector_result
        predicted += int(result.predicted_label != 0)
        available = [(i, p) for i, p in enumerate(board.people) if i not in rescued_before]
        in_range += int(any(np.linalg.norm(p.position - board.robot.position) <= board.sensor_range for _, p in available))
        if result.estimated_position is None:
            continue
        point = result.estimated_position
        distance = min((float(np.linalg.norm(p.position - point)) for _, p in available), default=float("inf"))
        rescued_distance = min((float(np.linalg.norm(board.people[i].position - point)) for i in rescued_before),
                               default=float("inf"))
        metal_distance = min((float(np.linalg.norm(m.position - point)) for m in board.metals), default=float("inf"))
        events.append({"step": index + 1, "time": board.time, "position": point.tolist(),
                       "distance_to_unrescued": distance if np.isfinite(distance) else None,
                       "near_unrescued": distance <= 6.5, "near_already_rescued_only": distance > 6.5 and rescued_distance <= 6.5,
                       "near_metal": metal_distance <= 6.5, "decision_mode": result.decision_mode,
                       "probability": result.person_probability})
        proposal_frames.append(index)
    if session.detector_error or session.coverage_error:
        raise RuntimeError({"detector": session.detector_error, "agent": session.coverage_error})
    proposals = len(events)
    correct = sum(e["near_unrescued"] for e in events)
    burst = max((sum(start <= frame < start + 25 for frame in proposal_frames) for start in proposal_frames), default=0)
    return {"setting": setting, "scenario": scenario, "environment": environment, "seed": seed,
            "requested_steps": steps, "steps": frames, "simulation_time": board.time, "initial_people": initial_people,
            "current_ma": current, "relay_gain": board.sensor.field_model.scaling["relay_gain"],
            "strength_factor": board.sensor.field_model.strength_factor, "sensor_range": board.sensor_range,
            "ping_proposals": proposals, "correct_proposals": correct, "false_proposals": proposals - correct,
            "unique_markers": len(board.detection_tracker.markers),
            "merged_marker_updates": sum(m.observations - 1 for m in board.detection_tracker.markers),
            "duplicates_near_rescued": sum(e["near_already_rescued_only"] for e in events),
            "false_near_metal": sum(not e["near_unrescued"] and e["near_metal"] for e in events),
            "max_proposals_25_steps": burst,
            "proposal_precision": correct / proposals if proposals else None,
            "proposals_per_100_steps": proposals / frames * 100 if frames else 0,
            "false_per_100_steps": (proposals - correct) / frames * 100 if frames else 0,
            "predicted_person_steps": predicted, "person_in_range_steps": in_range,
            "rescued": len(session.rescue_manager.rescued_indices), "path_length": path_length,
            "coverage": session.coverage_ratio, "wall_seconds": time.perf_counter() - started, "events": events}


def summarize(records: list[dict]) -> list[dict]:
    groups = {}
    for r in records:
        groups.setdefault((r["scenario"], r["setting"]), []).append(r)
    result = []
    for (scenario, setting), rows in sorted(groups.items()):
        steps = sum(r["steps"] for r in rows)
        proposals = sum(r["ping_proposals"] for r in rows)
        correct = sum(r["correct_proposals"] for r in rows)
        result.append({"scenario": scenario, "setting": setting, "runs": len(rows), "steps": steps,
                       "strength_factor": rows[0]["strength_factor"], "sensor_range": rows[0]["sensor_range"],
                       "ping_proposals": proposals, "correct_proposals": correct, "false_proposals": proposals - correct,
                       "unique_markers": sum(r["unique_markers"] for r in rows),
                       "proposal_precision": correct / proposals if proposals else None,
                       "false_per_100_steps": (proposals - correct) / steps * 100 if steps else 0,
                       "rescued": sum(r["rescued"] for r in rows),
                       "initial_people": sum(r["initial_people"] for r in rows),
                       "duplicates_near_rescued": sum(r["duplicates_near_rescued"] for r in rows),
                       "max_proposals_25_steps": max(r["max_proposals_25_steps"] for r in rows)})
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--output-prefix", default="parameter_ping_sweep")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 73])
    parser.add_argument("--settings", nargs="+", choices=list(SETTINGS), default=list(SETTINGS))
    parser.add_argument("--scenarios", nargs="+", choices=["regular", "no_people", "cluster_none"],
                        default=["regular", "no_people", "cluster_none"])
    args = parser.parse_args()
    from config_v2_1 import ENVIRONMENT_PRESETS
    root = Path(__file__).resolve().parents[1]
    output = root / "evaluation" / "results"
    weights = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in (root / "models").rglob("*.pth")}
    tasks = [(setting, scenario, env, seed, args.steps) for setting in args.settings
             for scenario in args.scenarios for env in ENVIRONMENT_PRESETS for seed in args.seeds]
    records = []
    started = time.perf_counter()
    partial = output / f"{args.output_prefix}_partial_v2_1.jsonl"
    with partial.open("w", encoding="utf-8") as stream, ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(run_case, task): task for task in tasks}
        for future in as_completed(jobs):
            result = future.result()
            records.append(result)
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            stream.flush()
            print(json.dumps({"completed": len(records), "total": len(tasks),
                              **{key: result[key] for key in ("setting", "scenario", "seed", "ping_proposals", "false_proposals", "unique_markers")}},
                             ensure_ascii=False), flush=True)
    summary = summarize(records)
    assert weights == {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in (root / "models").rglob("*.pth")}
    report = {"steps_limit": args.steps, "seeds": args.seeds, "cases": len(records),
              "total_steps": sum(r["steps"] for r in records), "wall_seconds": time.perf_counter() - started,
              "correctness_distance": 6.5, "metric": "proposals_within_6.5_of_not_yet_rescued_person_at_proposal_time",
              "limitations": ["synthetic_environments", "two_seeds_not_population_estimate",
                              "proximity_precision_not_identity_accuracy", "rescues_may_occur_later_at_old_pins"],
              "weights_unchanged": True, "weights": weights, "summary": summary,
              "records": sorted(records, key=lambda r: (r["scenario"], r["setting"], r["environment"], r["seed"]))}
    (output / f"{args.output_prefix}_v2_1.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output / f"{args.output_prefix}_v2_1.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(json.dumps({"completed": len(records), "steps": report["total_steps"], "summary": summary}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
