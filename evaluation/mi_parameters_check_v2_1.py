from __future__ import annotations

import ast
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from simulation.board_v2_1 import Board
from simulation.mi_calibration_v2_1 import MICalibration, DEFAULT_PROFILE
from simulation.coil_circuit_v2_1 import coil_field, MU0
from ui.mi_parameters_v2_1 import PARAMETERS, make_parameters, transfer_defaults
from web_runtime_v2_1 import WebSimulationSession


def run_checks() -> dict:
    root = Path(__file__).resolve().parents[1]
    files = list(root.rglob("*.py"))
    for path in files:
        ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        assert path.name == "__init__.py" or path.stem.endswith("_v2_1"), path
    original_bytes = DEFAULT_PROFILE.read_bytes()
    calibration = MICalibration()
    assert calibration.source == "provisional_theory"
    profile = calibration.profile
    manual = transfer_defaults(profile)
    values = {key: profile[key] * scale for key, _, scale in PARAMETERS}
    settings = make_parameters(profile, values, True, 50.0, [v * 2000 for _, v in manual["rows"]])
    custom = MICalibration(profile_overrides=settings)
    assert custom.source == "manual_assumption"
    assert custom.accepted_rows == 0
    np.testing.assert_allclose(custom.factors(np.array([0.2, 0.3, 0.6, 0.8])), 2.0, rtol=1e-7)
    np.testing.assert_equal(custom.factors(np.array([0.1, 1.0])), [1.0, 1.0])
    assert custom.status()["measured_range_m"] is None
    assert custom.status()["manual_range_m"] == [0.2, 0.8]
    for changes in ({"tx_current_rms_a": 0}, {"receiver_gain": -1}, {"receiver_output_noise_rms_v": float("nan")}):
        try:
            MICalibration(profile_overrides=changes)
        except ValueError:
            pass
        else:
            raise AssertionError(changes)
    a = Board(42)
    b = Board(42, mi_parameters={"tx_current_rms_a": profile["tx_current_rms_a"] * 2})
    a.people = b.people = []
    a.metals = a.metals[:1]
    b.metals = b.metals[:1]
    for board in (a, b):
        board.metals[0].position = board.robot.position + np.array([6.0, 0.0])
    av, ap = a.sensor.field_model.receive(a.robot.position[None], a)
    bv, bp = b.sensor.field_model.receive(b.robot.position[None], b)
    np.testing.assert_allclose(bv - bp, 2 * (av - ap), rtol=1e-7, atol=1e-14)
    r, n, current, z = 0.05, 100, 0.05, 0.3
    numerical = coil_field(np.array([[0.0, 0.0, z]]), np.zeros(3), np.array([0.0, 0.0, 1.0]), r, n, current, 256)
    expected = MU0 * n * current * r**2 / (2 * (r*r + z*z)**1.5)
    np.testing.assert_allclose(numerical[0, 2], expected, rtol=1e-12)
    with tempfile.TemporaryDirectory(prefix="mi-check-v2-1-") as folder:
        folder = Path(folder)
        test_profile = deepcopy(profile)
        test_profile.pop("input_mode", None)
        test_profile["manual_transfer"]["enabled"] = False
        test_profile["measurement_file"] = "measurements_v2_1.csv"
        (folder / "profile_v2_1.json").write_text(json.dumps(test_profile), encoding="utf-8")
        header = (root / "calibration" / "measurements_v2_1.csv").read_text(encoding="utf-8").splitlines()[0].split(",")
        with (folder / "measurements_v2_1.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=header)
            writer.writeheader()
            for i, d in enumerate((0.2, 0.4)):
                writer.writerow(dict(run_id=f"fixture_{i}", repeat_id=1, provenance="measured", medium="freshwater",
                                     frequency_hz=2000, distance_m=d, rx_angle_deg=0, relay_count=0,
                                     tx_current_rms_a=0.05, rx_voltage_rms_v=calibration.direct_transfer(d) * 0.05 * 172 * 0.8,
                                     voltage_location="amplifier_output", receiver_gain=172, valid="true"))
        measured = MICalibration(folder / "profile_v2_1.json")
        assert measured.source == "measured_transfer"
        np.testing.assert_allclose(measured.factors(np.array([0.2, 0.4])), 0.8)
    started = time.perf_counter()
    sessions = []
    for seed in (42, 73, 99):
        session = WebSimulationSession()
        session.reset(seed)
        session.robot_perception = True
        assert session.detector_error is None and session.coverage_error is None
        for _ in range(100):
            session.step()
            scan = session.board.magnetic_scan
            assert np.isfinite(scan.vectors).all()
        assert session.detector_error is None and session.coverage_error is None
        state = session.snapshot()
        assert "people" not in state and "metals" not in state
        assert state["actual_people"] is None
        sessions.append({"seed": seed, "steps": 100, "pins": state["marker_count"]})
    session.apply_command({"type": "mi_parameters", "value": {
        "tx_current_rms_a": 0.025,
        "manual_transfer": settings["manual_transfer"],
    }})
    assert session.mi_error is None and session.board.time == 0
    assert session.board.sensor.field_model.profile["tx_current_rms_a"] == 0.025
    assert not session.board.detection_tracker.markers
    assert len(session.detector_runtime.features) == 0
    for action in ("reset", "randomize"):
        session.apply_command({"type": action})
        assert session.board.sensor.field_model.profile["tx_current_rms_a"] == 0.025
        assert session.snapshot()["mi"]["source"] == "manual_assumption"
    old_board = session.board
    session.apply_command({"type": "mi_parameters", "value": {"tx_current_rms_a": -1}})
    assert session.mi_error and session.board is old_board
    assert DEFAULT_PROFILE.read_bytes() == original_bytes
    from desktop_app_v2_1 import DesktopSimulationApp
    app = DesktopSimulationApp()
    app.timer.stop()
    initial = app.board.robot.position.copy()
    app.apply_parameters(settings)
    np.testing.assert_equal(app.board.robot.position, initial)
    assert app.board.sensor.field_model.calibration.source == "manual_assumption"
    app.reset()
    assert app.board.sensor.field_model.calibration.source == "manual_assumption"
    app.open_parameters()
    from matplotlib.backend_bases import ResizeEvent
    canvas = app.parameter_window.figure.canvas
    canvas.callbacks.process("resize_event", ResizeEvent("resize_event", canvas))
    assert app.parameter_window.apply()
    output = root / "evaluation" / "results"
    output.mkdir(parents=True, exist_ok=True)
    app.figure.savefig(output / "mi_desktop_preview_v2_1.png", dpi=100)
    app.parameter_window.figure.savefig(output / "mi_parameters_preview_v2_1.png", dpi=100)
    plt.close("all")
    return {"syntax_files": len(files), "manual_curve": "PASS", "measured_csv": "PASS", "invalid_input": "PASS",
            "biot_savart_axis": "PASS", "tx_current_linearity": "PASS", "reset_randomize": "PASS",
            "observation_only": "PASS", "desktop_controls": "PASS", "sessions": sessions,
            "headless_seconds": round(time.perf_counter() - started, 3),
            "default_profile_unchanged": hashlib.sha256(original_bytes).hexdigest(),
            "accuracy_validation": "NOT_PERFORMED_V2_WEIGHTS_REUSED"}


if __name__ == "__main__":
    results = run_checks()
    target = Path(__file__).resolve().parent / "results" / "mi_parameters_check_v2_1.json"
    target.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))
