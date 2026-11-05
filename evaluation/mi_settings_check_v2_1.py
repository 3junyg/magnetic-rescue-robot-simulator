from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np

from desktop_app_v2_1 import DesktopSimulationApp
from simulation.mi_calibration_v2_1 import MICalibration, DEFAULT_PROFILE
from simulation.mi_settings_v2_1 import default_settings, design_values, expected_voltage_mv, make_settings
from web_runtime_v2_1 import WebSimulationSession


def main() -> None:
    original = DEFAULT_PROFILE.read_bytes()
    defaults = default_settings()
    values = design_values(defaults)
    np.testing.assert_allclose(expected_voltage_mv(defaults, 30), 2.650750183219038, rtol=1e-10)
    theory = make_settings(defaults, {**values, "tx_current_rms_a": 25}, "예상값", "coil_input",
                           [20, 30, 40, 50, 60, 80], [None] * 6)
    assert theory["input_mode"] == "theory" and not theory["manual_transfer"]["enabled"]
    np.testing.assert_allclose(theory["manual_transfer"]["rows"][1][1] * 1000, 1.325375091609519, rtol=1e-10)
    np.testing.assert_allclose(expected_voltage_mv({**defaults, "tx_turns": 200}, 30),
                               expected_voltage_mv(defaults, 30) * 2)
    without_csv = MICalibration(profile_overrides={**theory, "measurement_file": "does_not_exist_v2_1.csv"})
    assert without_csv.source == "provisional_theory"
    voltages = [expected_voltage_mv(defaults, d, "amplifier_output") * 0.8 for d in (25, 45)] + [None] * 4
    measured = make_settings(defaults, values, "실험값", "amplifier_output", [25, 45, 50, 60, 70, 80], voltages)
    calibration = MICalibration(profile_overrides=measured)
    assert calibration.source == "measured_transfer"
    np.testing.assert_allclose(calibration.correction, 0.8)
    for distances, result in (([20, 30], [1, None]), ([20, 20], [1, 2])):
        try:
            make_settings(defaults, values, "실험값", "coil_input", distances, result)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid experiment accepted")
    app = DesktopSimulationApp()
    app.timer.stop()
    app.open_parameters()
    window = app.parameter_window
    assert sum(isinstance(value, Button) for value in vars(window).values()) == 2
    assert not hasattr(window, "auto_apply_timer")
    assert all(not box.active for box in window.voltages)
    before = app.board
    window.boxes["tx_current_rms_a"].set_val("25")
    assert app.board is before
    np.testing.assert_allclose(float(window.voltages[1].text), 1.325375091609519, rtol=1e-5)
    assert window.apply()
    assert app.board is not before
    assert app.board.sensor.field_model.profile["tx_current_rms_a"] == 0.025
    window.mode_radio.set_active(1)
    assert all(not box.text for box in window.voltages)
    assert all(box.active for box in window.voltages)
    window.boxes["tx_current_rms_a"].set_val("50")
    assert not hasattr(window, "location_radio")
    assert "receiver_gain" not in window.boxes
    for i in range(2):
        window.distances[i].set_val(str((25, 45)[i]))
        window.voltages[i].set_val(str(voltages[i]))
    before = app.board
    assert app.board.sensor.field_model.calibration.source == "provisional_theory"
    assert window.apply()
    assert app.board is not before
    assert app.board.sensor.field_model.calibration.source == "measured_transfer"
    np.testing.assert_allclose(app.board.sensor.field_model.calibration.correction, 0.8)
    assert window.reset_to_defaults()
    assert window.mode == "예상값" and all(not box.active for box in window.voltages)
    assert design_values(app.board.sensor.field_model.profile) == values
    assert app.board.sensor.field_model.calibration.source == "provisional_theory"
    assert app.board.time == 0
    assert not app.board.detection_tracker.markers
    assert len(app.detector.features) == 0
    session = WebSimulationSession()
    initial_revision = session.mi_revision
    session.apply_command({"type": "mi_settings", "value": {
        "design": values, "mode": "실험값", "location": "amplifier_output",
        "distances_cm": [25, 45, 50, 60, 70, 80], "voltages_mv": voltages,
    }})
    assert session.mi_error is None
    assert session.mi_revision == initial_revision + 1
    assert session.snapshot()["mi"]["mode"] == "실험값"
    for _ in range(32):
        session.step()
    assert session.detector_error is None
    session.apply_command({"type": "reset_mi_settings"})
    assert session.snapshot()["mi"]["mode"] == "예상값"
    assert session.snapshot()["mi"]["design"] == values
    assert session.mi_revision == initial_revision + 2
    assert DEFAULT_PROFILE.read_bytes() == original
    output = Path(__file__).resolve().parent / "results"
    output.mkdir(parents=True, exist_ok=True)
    window.figure.savefig(output / "mi_settings_two_modes_v2_1.png", dpi=100)
    app.figure.savefig(output / "mi_settings_desktop_v2_1.png", dpi=100)
    plt.close("all")
    result = {"two_modes": "PASS", "theory_computes_results": "PASS", "theory_ignores_csv_and_old_measurements": "PASS",
              "measured_input_and_gain_conversion": "PASS", "no_auto_apply": "PASS", "exactly_two_buttons": "PASS",
              "full_settings_reset": "PASS", "runtime_apply_reset": "PASS", "default_file_unchanged": "PASS",
              "real_measurements_used_in_test": False, "accuracy_validation": "NOT_PERFORMED"}
    (output / "mi_settings_check_v2_1.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
