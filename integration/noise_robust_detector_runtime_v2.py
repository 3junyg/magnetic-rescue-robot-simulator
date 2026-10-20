from pathlib import Path

from integration.human_detector_runtime_v2 import HumanDetectorRuntime


class NoiseRobustHumanDetectorRuntime(HumanDetectorRuntime):
    def __init__(self, model_path: str | Path | None = None, device: str = "cpu", add_markers: bool = True) -> None:
        root = Path(__file__).resolve().parents[1]
        selected = Path(model_path) if model_path else root / "models" / "human_detector_noise_robust" / "best_human_transition_detector_v2.pth"
        super().__init__(selected, device, add_markers)
        self.presence_threshold = 0.50
        self.marker_probability_threshold = 0.60
        self.location_threshold = 0.375
