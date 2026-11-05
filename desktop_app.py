import matplotlib.pyplot as plt
from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider

from config import DEFAULT_SEED, ENVIRONMENT_PRESETS, TIME_STEP
from integration.controlled_stepper import ControlledBoardStepper
from integration.coverage_board_runtime import CoverageBoardRuntime
from integration.human_detector_runtime import HumanDetectorRuntime, HumanDetectorResult
from integration.rescue_manager import RescueManager
from simulation.board import Board
from ui.renderer import draw_board


class DesktopSimulationApp:
    def __init__(self) -> None:
        self.seed = DEFAULT_SEED
        self.environment = next(iter(ENVIRONMENT_PRESETS))
        self.noise_level = float(ENVIRONMENT_PRESETS[self.environment]["noise"])
        self.board = Board(self.seed, self.environment, self.noise_level)
        self.running = False
        self.robot_perception = False
        self.navigation_mode = "Manual"
        self.stepper = ControlledBoardStepper()
        self.rescue_manager = RescueManager()
        try:
            self.detector = HumanDetectorRuntime()
            self.detector_error = ""
        except Exception as error:
            self.detector = None
            self.detector_error = str(error)
        try:
            self.coverage = CoverageBoardRuntime()
            self.coverage_error = ""
        except Exception as error:
            self.coverage = None
            self.coverage_error = str(error)
        self.detector_result = HumanDetectorResult(0.0, 0, "WARMING UP", 0.0, None, 0, 0)
        self.last_action = "stop"
        self.coverage_ratio = 0.0
        self.figure = plt.figure(figsize=(14, 8))
        self.map_axis = self.figure.add_axes([0.05, 0.08, 0.72, 0.86])
        self.info_axis = self.figure.add_axes([0.80, 0.55, 0.18, 0.18])
        self.info_axis.axis("off")
        self.start_button = Button(self.figure.add_axes([0.80, 0.46, 0.08, 0.055]), "Start")
        self.stop_button = Button(self.figure.add_axes([0.90, 0.46, 0.08, 0.055]), "Stop")
        self.reset_button = Button(self.figure.add_axes([0.80, 0.38, 0.08, 0.055]), "Reset")
        self.random_button = Button(self.figure.add_axes([0.90, 0.38, 0.08, 0.055]), "Randomize")
        self.truth_button = CheckButtons(self.figure.add_axes([0.80, 0.29, 0.18, 0.06]), ["Robot Perception"], [False])
        self.environment_button = RadioButtons(self.figure.add_axes([0.80, 0.07, 0.18, 0.18]), list(ENVIRONMENT_PRESETS), active=0)
        self.mode_button = RadioButtons(self.figure.add_axes([0.80, 0.68, 0.18, 0.12]), ["Manual", "CNN Agent"], active=0)
        self.noise_slider = Slider(self.figure.add_axes([0.80, 0.015, 0.18, 0.025]), "Noise", 0.0, 5.0, valinit=self.noise_level, valstep=0.01)
        self.start_button.on_clicked(self.start)
        self.stop_button.on_clicked(self.stop)
        self.reset_button.on_clicked(self.reset)
        self.random_button.on_clicked(self.randomize)
        self.truth_button.on_clicked(self.toggle_perception)
        self.environment_button.on_clicked(self.change_environment)
        self.mode_button.on_clicked(self.change_mode)
        self.noise_slider.on_changed(self.change_noise_level)
        self.timer = self.figure.canvas.new_timer(interval=60)
        self.timer.add_callback(self.tick)
        self.timer.start()
        self.redraw()

    def start(self, event=None) -> None:
        self.running = True
        self.redraw()

    def stop(self, event=None) -> None:
        self.running = False
        self.redraw()

    def reset(self, event=None) -> None:
        self.board = Board(self.seed, self.environment, self.noise_level)
        self.running = False
        self._reset_models()
        self.redraw()

    def randomize(self, event=None) -> None:
        self.seed += 1
        self.board = Board(self.seed, self.environment, self.noise_level)
        self.running = False
        self._reset_models()
        self.redraw()

    def toggle_perception(self, label=None) -> None:
        self.robot_perception = not self.robot_perception
        self.redraw()

    def change_environment(self, label: str) -> None:
        self.environment = label
        self.board = Board(self.seed, self.environment, self.noise_level)
        self.running = False
        self._reset_models()
        self.redraw()

    def _reset_models(self) -> None:
        self.stepper.reset()
        if self.detector is not None:
            self.detector.reset()
        self.rescue_manager.reset()
        if self.coverage is not None:
            self.coverage.reset()
        self.detector_result = HumanDetectorResult(0.0, 0, "WARMING UP", self.board.time, None, 0, 0)
        self.last_action = "stop"
        self.coverage_ratio = 0.0

    def change_mode(self, label: str) -> None:
        self.navigation_mode = label
        self.redraw()

    def change_noise_level(self, value: float) -> None:
        self.noise_level = float(value)
        self.board.set_enhanced_noise_level(self.noise_level)
        self.redraw()

    def tick(self) -> None:
        if self.running:
            for _ in range(2):
                if self.detector is not None and self.detector_result.timestamp != self.board.time:
                    self.detector_result = self.detector.process_board(self.board)
                    rescue_event = self.rescue_manager.process(self.board, self.detector_result)
                    if rescue_event is not None:
                        self.detector.ignore_position(rescue_event.position)
                if self.rescue_manager.all_rescued(self.board):
                    self.running = False
                    self.last_action = "stop"
                    break
                if self.navigation_mode == "CNN Agent" and self.coverage is not None:
                    self.last_action, self.coverage_ratio = self.coverage.decide(self.board)
                    self.stepper.step(self.board, self.last_action, 0.4)
                else:
                    self.last_action = "forward"
                    self.board.step(TIME_STEP)
                    self.coverage_ratio = float(self.board.perception.observed.mean())
                if self.detector is not None:
                    self.detector_result = self.detector.process_board(self.board)
                    rescue_event = self.rescue_manager.process(self.board, self.detector_result)
                    if rescue_event is not None:
                        self.detector.ignore_position(rescue_event.position)
                if self.rescue_manager.all_rescued(self.board):
                    self.running = False
                    self.last_action = "stop"
                    break
            self.redraw()

    def redraw(self) -> None:
        self.map_axis.clear()
        draw_board(self.map_axis, self.board, self.robot_perception)
        measurement = self.board.measurements[-1]
        distances = self.board.perception.candidate_distances(self.board.robot.position)
        nearest_candidate = f"{min(distances):.2f}" if len(distances) else "None"
        self.info_axis.clear()
        self.info_axis.axis("off")
        status = "RUNNING" if self.running else "STOPPED"
        text = (
            f"Status: {status}\n"
            f"Seed: {self.seed}\n"
            f"Time: {self.board.time:.1f} s\n"
            f"Noise level: {self.noise_level:.1f}\n"
            f"Bx: {measurement.bx:.3f}\n"
            f"By: {measurement.by:.3f}\n"
            f"Bz: {measurement.bz:.3f}\n"
            f"|B|: {measurement.magnitude:.3f}\n"
            f"Nearest metal candidate: {nearest_candidate}\n"
            f"Mode: {self.navigation_mode}\n"
            f"Action: {self.last_action}\n"
            f"Detector: {self.detector_result.status}\n"
            f"Person probability: {self.detector_result.person_probability * 100:.1f}%\n"
            f"Prediction: {'PERSON' if self.detector_result.predicted_label else 'NO PERSON'}\n"
            f"Observed coverage: {self.coverage_ratio * 100:.1f}%\n"
            f"Rescued: {len(self.rescue_manager.rescued_indices)}"
        )
        if self.rescue_manager.all_rescued(self.board):
            text += "\nSearch state: COMPLETE"
        elif self.coverage is not None and self.coverage.patrol_mode:
            text += "\nSearch state: PATROLLING"
        else:
            text += "\nSearch state: EXPLORING"
        if self.detector_error:
            text += f"\nDetector error: {self.detector_error}"
        if self.coverage_error:
            text += f"\nCoverage error: {self.coverage_error}"
        self.info_axis.text(0.0, 1.0, text, va="top", fontsize=11)
        self.figure.canvas.draw_idle()

    def show(self) -> None:
        plt.show()


if __name__ == "__main__":
    DesktopSimulationApp().show()
