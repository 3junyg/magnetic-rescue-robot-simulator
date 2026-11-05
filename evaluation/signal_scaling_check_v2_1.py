from __future__ import annotations

import ast
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from desktop_app_v2_1 import DesktopSimulationApp
from integration.human_detector_runtime_v2_1 import HumanDetectorRuntime
from ml.features_v2_1 import extract_scan_features
from simulation.board_v2_1 import Board
from simulation.mi_settings_v2_1 import default_settings, design_values, make_settings
from simulation.relay_settings_v2_1 import with_relay
from ui.renderer_v2_1 import draw_board
from smooth_perception_renderer_v2_1 import draw_smooth_perception
from web_runtime_v2_1 import WebSimulationSession


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    for path in root.rglob("*.py"):
        ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    default = default_settings()
    values = design_values(default)
    profiles = [make_settings(default, {**values, "tx_current_rms_a": current}, "예상값", "coil_input", [30], [None])
                for current in (25, 50, 100)]
    boards = [Board(42, mi_parameters=profile) for profile in profiles]
    np.testing.assert_allclose([b.sensor.field_model.strength_factor for b in boards], [0.5, 1, 2])
    np.testing.assert_allclose([b.sensor_range for b in boards], 16 * np.cbrt([0.5, 1, 2]))
    assert [len(b.magnetic_scan.positions) for b in boards] == sorted(len(b.magnetic_scan.positions) for b in boards)
    bg = np.array([24., -7., 39.])
    for kind in ("metal", "human"):
        fields = []
        for b in boards:
            if kind == "metal":
                b.people = []
                b.metals = b.metals[:1]
                b.metals[0].position = b.robot.position + [3, 0]
            else:
                b.metals = []
                if not b.people:
                    reference = Board(42)
                    b.people = reference.people[:1]
                b.people = b.people[:1]
                b.people[0].position = b.robot.position + [3, 0]
            b.time = 0.7
            fields.append(b.sensor.field_model.field_at(b.robot.position, b) - bg)
        np.testing.assert_allclose(fields[0], fields[1] * 0.5, atol=1e-13)
        np.testing.assert_allclose(fields[2], fields[1] * 2, atol=1e-13)
    for b in boards:
        assert b.range_scan.sensor_range == b.sensor_range
        assert b.magnetic_scan.sensor_range == b.sensor_range
        assert np.all(np.linalg.norm(b.magnetic_scan.positions - b.robot.position, axis=1) <= b.sensor_range)
        assert np.max(b.range_scan.distances) <= b.sensor_range
        assert extract_scan_features(b.magnetic_scan, b.robot.position, b.robot.heading).shape == (4, 8, 8)
        runtime = HumanDetectorRuntime()
        runtime.process_observation(b.observation())
        assert runtime.sensor_range == b.sensor_range
        point = runtime._location_to_world(0, b.robot.position, 0)
        np.testing.assert_allclose(np.linalg.norm(point - b.robot.position), b.sensor_range / 8)
    measured = make_settings(default, values, "실험값", "coil_input", [30, 40],
                            [2.650750183219038 * 2, 1.157046276339752 * 2])
    calibrated = Board(42, mi_parameters=measured)
    assert calibrated.sensor.field_model.strength_factor > 1.9
    assert not calibrated.sensor.field_model.uses_physical_mi
    relay = Board(42, mi_parameters=with_relay(default, {"enabled": True, "count": 1, "spacing_cm": 30., "axis": "Y"}))
    np.testing.assert_allclose(relay.sensor.field_model.strength_factor, relay.sensor.field_model.relay_summary["gain"])
    app = DesktopSimulationApp()
    app.timer.stop()
    app.open_parameters()
    window = app.parameter_window
    assert not hasattr(window, "model_toggle")
    original = app.board
    window.boxes["tx_current_rms_a"].set_val("100")
    assert app.board is original
    assert window.apply()
    np.testing.assert_allclose(app.board.sensor.field_model.strength_factor, 2)
    assert app.board.sensor_range > 16
    figure, axes = plt.subplots(1, 2, figsize=(15, 6))
    draw_board(axes[0], app.board, False)
    draw_smooth_perception(axes[1], app.board, True)
    for axis in axes:
        assert any(np.isclose(getattr(patch, "radius", -1), app.board.sensor_range) for patch in axis.patches)
    output = root / "evaluation" / "results"
    figure.savefig(output / "signal_scaling_preview_v2_1.png", dpi=110)
    assert window.reset_to_defaults()
    assert app.board.sensor_range == 16 and app.board.sensor.field_model.strength_factor == 1
    runs = []
    for seed, profile in zip((42, 73, 99), profiles):
        session = WebSimulationSession()
        session.reset(seed, mi_parameters=profile)
        for _ in range(100):
            session.step()
            assert np.isfinite(session.board.magnetic_scan.vectors).all()
        assert session.detector_error is None and session.coverage_error is None
        assert session.detector_result.status != "UNCALIBRATED"
        state = session.snapshot()
        assert state["board"]["sensor_range"] == session.board.sensor_range
        runs.append({"seed": seed, "steps": 100, "strength": state["mi"]["scaling"]["strength_factor"],
                     "range": session.board.sensor_range, "markers": state["marker_count"]})
    plt.close("all")
    result = {"scope": "amplitude_and_range_only", "both_materials_same_factor": "PASS", "range_scan_features_render": "PASS",
              "measured_and_relay_inputs": "PASS", "apply_reset": "PASS", "runs": runs,
              "range_relation": "synthetic_cube_root_not_empirical", "accuracy_improvement": "NOT_CLAIMED"}
    (output / "signal_scaling_check_v2_1.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
