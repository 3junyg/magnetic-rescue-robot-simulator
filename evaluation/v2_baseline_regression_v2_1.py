from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


RUNNER = r"""
import hashlib, json, importlib, sys
version, seed, environment, steps = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4])
Session = importlib.import_module("web_runtime_" + version).WebSimulationSession
s = Session()
s.reset(seed, environment)
digest = hashlib.sha256()
for _ in range(steps):
    s.step()
    b = s.board
    m = b.measurements[-1]
    row = dict(time=b.time, sensor=[m.bx,m.by,m.bz,m.magnitude], action=s.last_action,
               robot=b.robot.position.tolist(), heading=b.robot.heading,
               probability=s.detector_result.person_probability, prediction=s.detector_result.predicted_label,
               markers=[[m.position.tolist(),m.confidence,m.timestamp,m.observations] for m in b.detection_tracker.markers],
               rescued=sorted(s.rescue_manager.rescued_indices), coverage=s.coverage_ratio)
    digest.update(json.dumps(row,sort_keys=True,separators=(',',':')).encode())
    digest.update(b.magnetic_scan.vectors.tobytes())
assert s.detector_error is None and s.coverage_error is None
print(json.dumps(dict(digest=digest.hexdigest(),markers=len(s.board.detection_tracker.markers),
                     rescued=len(s.rescue_manager.rescued_indices),time=s.board.time)))
"""


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    v2 = root.with_name(root.name.replace("_v2.1", "_v2"))
    if not (v2 / "web_runtime_v2.py").exists():
        raise FileNotFoundError(f"V2 baseline not found: {v2}")
    results = []
    for seed, environment in ((42, "도심 침수 지역"), (73, "산업 시설"), (99, "주거 지역")):
        outputs = []
        for version, folder in (("v2", v2), ("v2_1", root)):
            output = subprocess.check_output([sys.executable, "-B", "-c", RUNNER, version, str(seed), environment, "600"],
                                             cwd=folder, encoding="utf-8")
            outputs.append(json.loads(output))
        assert outputs[0] == outputs[1], {"seed": seed, "v2": outputs[0], "v2.1": outputs[1]}
        results.append({"seed": seed, "environment": environment, "steps_each": 600, "exact_match": True, **outputs[0]})
        print(json.dumps(results[-1], ensure_ascii=False), flush=True)
    path = root / "evaluation" / "results" / "v2_baseline_regression_v2_1.json"
    path.write_text(json.dumps({"mode": "v2", "comparison": "sensor_scan_robot_probability_markers_rescue_every_step",
                               "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
