import argparse

import desktop_app_v2
from train_enhanced_noise_detector_v2 import EnhancedNoiseBoard


desktop_app_v2.Board = EnhancedNoiseBoard


class EnhancedNoiseSimulationApp(desktop_app_v2.DesktopSimulationApp):
    def redraw(self) -> None:
        super().redraw()
        level = getattr(self.board, "enhanced_noise_level", 0.0)
        self.figure.suptitle(
            f"Enhanced Environmental Noise Simulation | noise level: {level:.2f}",
            fontsize=13,
        )
        self.figure.canvas.draw_idle()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--noise-level", type=float, default=1.4)
    args = parser.parse_args()
    EnhancedNoiseBoard.noise_level_min = max(0.0, args.noise_level)
    EnhancedNoiseBoard.noise_level_max = EnhancedNoiseBoard.noise_level_min
    EnhancedNoiseSimulationApp().show()


if __name__ == "__main__":
    main()
