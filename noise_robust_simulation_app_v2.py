import argparse

import desktop_app_v2
from matplotlib.widgets import Slider
from integration.noise_robust_detector_runtime_v2 import NoiseRobustHumanDetectorRuntime
from train_enhanced_noise_detector_v2 import EnhancedNoiseBoard


desktop_app_v2.Board = EnhancedNoiseBoard
desktop_app_v2.HumanDetectorRuntime = NoiseRobustHumanDetectorRuntime


class NoiseRobustSimulationApp(desktop_app_v2.DesktopSimulationApp):
    def __init__(self) -> None:
        super().__init__()
        self.noise_slider = Slider(
            self.figure.add_axes([0.80, 0.015, 0.18, 0.025]),
            "Noise",
            0.0,
            5.0,
            valinit=float(self.board.enhanced_noise_level),
            valstep=0.1,
        )
        self.noise_slider.on_changed(self.change_noise_level)

    def change_noise_level(self, value: float) -> None:
        level = max(0.0, min(5.0, float(value)))
        EnhancedNoiseBoard.noise_level_min = level
        EnhancedNoiseBoard.noise_level_max = level
        self.board.enhanced_noise_level = level
        self.board.sensor.set_level(level)
        self.redraw()

    def redraw(self) -> None:
        super().redraw()
        level = getattr(self.board, "enhanced_noise_level", 0.0)
        self.figure.suptitle(f"Noise-Robust Human Detector | environmental noise: {level:.2f}", fontsize=13)
        self.figure.canvas.draw_idle()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--noise-level", type=float, default=5.0)
    args = parser.parse_args()
    EnhancedNoiseBoard.noise_level_min = max(0.0, args.noise_level)
    EnhancedNoiseBoard.noise_level_max = EnhancedNoiseBoard.noise_level_min
    NoiseRobustSimulationApp().show()


if __name__ == "__main__":
    main()
