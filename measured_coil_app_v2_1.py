import numpy as np

from desktop_app_v2_1 import DesktopSimulationApp
from simulation.coil_circuit_v2_1 import coil_field
from simulation.mi_settings_v2_1 import default_settings


# Single-coil axis medians minus the empty-test medians, in µT.
COIL_DELTAS = {
    10: (3.81, 3.20, -5.47),
    20: (3.20, 1.30, -4.86),
    30: (2.51, -0.11, -2.35),
    40: (2.51, 0.50, -0.75),
}

# Median per-axis robust scales of time-detrended three-sample residuals
# across in-out, left-right, and both rotation recordings, in µT.
MOTION_RESIDUAL_SCALE_UT = np.array([0.39731551, 0.32500220, 0.97673064])


def fitted_strength(profile: dict) -> float:
    distances_m = np.asarray(list(COIL_DELTAS)) / 100.0
    points = np.column_stack((np.zeros((len(distances_m), 2)), distances_m))
    fields = coil_field(points, np.zeros(3), np.array([0., 0., 1.]),
                        profile["tx_radius_m"], profile["tx_turns"], profile["tx_current_rms_a"],
                        profile["coil_segments"], profile["wire_core_m"])
    expected_ut = np.linalg.norm(fields, axis=1) * 1e6
    measured_ut = np.linalg.norm(list(COIL_DELTAS.values()), axis=1)
    return float(np.dot(expected_ut, measured_ut) / np.dot(expected_ut, expected_ut))


class MeasuredCoilApp(DesktopSimulationApp):
    def __init__(self) -> None:
        super().__init__()
        for button in (self.parameter_button, self.relay_button):
            button.set_active(False)
            button.ax.set_visible(False)
        self.figure.canvas.manager.set_window_title("V2.1 실측 보정 시뮬레이션")
        profile = default_settings()
        factor = fitted_strength(profile)
        profile["coil_layout_strength_factor"] = factor
        profile["stationary_metal_scan"] = True
        noise_shape = MOTION_RESIDUAL_SCALE_UT / np.sqrt(np.mean(MOTION_RESIDUAL_SCALE_UT ** 2))
        profile["sensor_noise_axis_scale"] = noise_shape.tolist()
        self.apply_parameters(profile)
        self.figure.suptitle(
            f"단일 코일 실측 보정 | 기존 코일 모델 대비 {factor:.3f}배 | "
            f"모델 센서 범위 {self.board.sensor_range:.2f}\n"
            f"이동·회전 실측 기반 축별 잡음 배율 (X/Y/Z): {noise_shape[0]:.2f} / {noise_shape[1]:.2f} / {noise_shape[2]:.2f}"
        )
        self.figure.canvas.draw_idle()


if __name__ == "__main__":
    MeasuredCoilApp().show()
