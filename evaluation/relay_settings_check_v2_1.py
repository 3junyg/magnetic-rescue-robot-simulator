from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backend_bases import ResizeEvent
import numpy as np

from desktop_app_v2_1 import DesktopSimulationApp
from simulation.board_v2_1 import Board
from simulation.coil_circuit_v2_1 import coaxial_mutual
from simulation.mi_settings_v2_1 import default_settings
from simulation.relay_settings_v2_1 import relay_summary, with_relay
from web_runtime_v2_1 import WebSimulationSession


def main() -> None:
    profile = default_settings()
    off = relay_summary(profile)
    assert off["ratio_real"] == 1 and off["ratio_imag"] == 0
    active = {"enabled": True, "count": 1, "spacing_cm": 30.0, "axis": "Y"}
    enabled = with_relay({**profile, "signal_mode": "calibrated_mi"}, active)
    summary = relay_summary(enabled)
    relay = profile["relay_axes"]["Y"]
    omega = 2 * np.pi * profile["frequency_hz"]
    mutual = sum(coaxial_mutual(relay["radius_m"], profile["tx_radius_m"],
                              abs(0.3 + offset), relay["turns_per_coil"], profile["tx_turns"])
                 for offset in (-relay["separation_m"] / 2, relay["separation_m"] / 2))
    direct = coaxial_mutual(profile["tx_radius_m"], profile["rx_radius_m"], 0.6,
                            profile["tx_turns"], profile["rx_turns"])
    impedance = relay["resistance_ohm"] + 1j * (omega * relay["inductance_h"] - 1 / (omega * relay["capacitance_f"]))
    analytic_ratio = 1 - 1j * omega * mutual**2 / impedance / direct
    ratio = complex(summary["ratio_real"], summary["ratio_imag"])
    np.testing.assert_allclose(ratio, analytic_ratio, rtol=1e-12)
    doubled = relay_summary({**enabled, "tx_current_rms_a": profile["tx_current_rms_a"] * 2})
    np.testing.assert_allclose(doubled["input_rms_v"], summary["input_rms_v"] * 2)
    np.testing.assert_allclose(doubled["gain"], summary["gain"])
    assert not np.isclose(summary["gain"], relay_summary(with_relay(profile, {**active, "spacing_cm": 40}))["gain"])
    assert not np.isclose(summary["gain"], relay_summary(with_relay(profile, {**active, "count": 2}))["gain"])
    for bad in ({"count": 1.2}, {"count": 9}, {"spacing_cm": 0}, {"spacing_cm": float("nan")},
                {"axis": "invalid"}, {"enabled": True, "spacing_cm": 1}):
        try:
            with_relay(profile, {**active, **bad})
        except (ValueError, TypeError):
            pass
        else:
            raise AssertionError(bad)
    old = {**deepcopy(profile), "signal_mode": "calibrated_mi"}
    old.update(receiver_gain=172.0, receiver_output_noise_rms_v=0.005, receiver_max_ac_rms_v=1.6)
    before, after = Board(42, mi_parameters=old), Board(42, mi_parameters={**profile, "signal_mode": "calibrated_mi"})
    for name in ("bx", "by", "bz", "magnitude"):
        np.testing.assert_allclose(getattr(before.measurements[-1], name), getattr(after.measurements[-1], name), rtol=1e-12)
    np.testing.assert_allclose(before.magnetic_scan.vectors, after.magnetic_scan.vectors, rtol=1e-12, atol=1e-12)
    a, b = Board(42, mi_parameters=profile), Board(42, mi_parameters=enabled)
    total_a, reference_a = a.sensor.field_model.receive(a.robot.position[None], a)
    total_b, reference_b = b.sensor.field_model.receive(b.robot.position[None], b)
    np.testing.assert_allclose(total_b - reference_b, ratio * (total_a - reference_a), rtol=1e-10, atol=1e-15)
    app = DesktopSimulationApp()
    app.timer.stop()
    app.open_relay_parameters()
    window = app.relay_window
    board = app.board
    window.enabled.set_active(1)
    assert app.board is board
    assert window.apply()
    assert app.board is not board and app.board.time == 0
    np.testing.assert_allclose(app.board.sensor.field_model.relay_factor, ratio)
    assert not app.board.detection_tracker.markers and len(app.detector.features) == 0
    app.open_parameters()
    assert not hasattr(app.parameter_window, "location_radio")
    assert "receiver_gain" not in app.parameter_window.boxes
    app.parameter_window.boxes["tx_current_rms_a"].set_val("25")
    assert app.parameter_window.apply()
    assert app.board.sensor.field_model.profile["relay_settings"]["enabled"]
    canvas = window.figure.canvas
    canvas.callbacks.process("resize_event", ResizeEvent("resize_event", canvas))
    output = Path(__file__).resolve().parent / "results"
    output.mkdir(exist_ok=True)
    window.figure.savefig(output / "relay_settings_preview_v2_1.png", dpi=100)
    assert app.parameter_window.reset_to_defaults()
    assert not app.board.sensor.field_model.profile["relay_settings"]["enabled"]
    runs = []
    for seed, count, axis in ((42, 1, "Y"), (73, 3, "X"), (99, 8, "Z")):
        session = WebSimulationSession()
        session.reset(seed, mi_parameters={**profile, "signal_mode": "calibrated_mi"})
        original = session.board
        settings = {**active, "count": count, "axis": axis}
        session.apply_command({"type": "relay_preview", "value": {"settings": settings, "request_id": seed}})
        assert session.board is original and session.relay_preview["error"] is None
        session.apply_command({"type": "relay_settings", "value": settings})
        assert session.mi_error is None and session.board is not original
        for _ in range(100):
            session.step()
            assert np.isfinite(session.board.magnetic_scan.vectors).all()
        assert session.detector_error is None and session.coverage_error is None
        assert not session.board.detection_tracker.markers
        assert not session.rescue_manager.rescued_indices
        assert session.detector_result.status == "UNCALIBRATED"
        session.apply_command({"type": "randomize"})
        assert session.board.sensor.field_model.profile["relay_settings"] == settings
        session.apply_command({"type": "reset_mi_settings"})
        assert session.board.sensor.field_model.relay_factor == 1
        runs.append({"seed": seed, "relay_count": count, "axis": axis, "steps": 100})
    plt.close("all")
    result = {"single_relay_analytic": "PASS", "spacing_count_current": "PASS", "invalid_settings": "PASS",
              "disabled_sensor_baseline": "PASS", "actual_signal_integration": "PASS", "desktop_apply_reset": "PASS",
              "headless_runs": runs, "model": "coaxial_equal_spacing_equivalent_link", "real_measurements": False}
    (output / "relay_settings_check_v2_1.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
