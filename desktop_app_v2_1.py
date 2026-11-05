import matplotlib.pyplot as plt
from matplotlib.widgets import Button, CheckButtons, RadioButtons

from config_v2_1 import DEFAULT_SEED, ENVIRONMENT_PRESETS, TIME_STEP
from integration.controlled_stepper_v2_1 import ControlledBoardStepper
from integration.coverage_board_runtime_v2_1 import CoverageBoardRuntime
from integration.human_detector_runtime_v2_1 import HumanDetectorRuntime, HumanDetectorResult
from integration.rescue_manager_v2_1 import RescueManager
from simulation.board_v2_1 import Board
from ui.renderer_v2_1 import draw_board
from ui.patrol_overlay_v2_1 import draw_patrol_overlay
from ui.mi_parameters_v2_1 import MIParameterWindow, SOURCE_LABELS
from ui.relay_parameters_v2_1 import RelayParameterWindow


class DesktopSimulationApp:
    def __init__(self) -> None:
        self.seed = DEFAULT_SEED
        self.environment = next(iter(ENVIRONMENT_PRESETS))
        self.mi_parameters = None
        self.parameter_window = None
        self.relay_window = None
        self.board = Board(self.seed, self.environment)
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
        self.figure = plt.figure(figsize=(15, 9))
        self.figure.canvas.manager.set_window_title("인명 탐지 로봇 시뮬레이션 V2.1")
        self.map_axis = self.figure.add_axes([0.05, 0.08, 0.72, 0.86])
        self.info_axis = self.figure.add_axes([0.80, 0.50, 0.19, 0.37])
        self.info_axis.axis("off")
        self.parameter_button = Button(self.figure.add_axes([0.80, 0.945, 0.18, 0.035]), "자기장 설정")
        self.parameter_button.on_clicked(self.open_parameters)
        self.relay_button = Button(self.figure.add_axes([0.80, 0.90, 0.18, 0.035]), "코일 중계 설정")
        self.relay_button.on_clicked(self.open_relay_parameters)
        self.start_button = Button(self.figure.add_axes([0.80, 0.435, 0.08, 0.05]), "실행")
        self.stop_button = Button(self.figure.add_axes([0.90, 0.435, 0.08, 0.05]), "정지")
        self.reset_button = Button(self.figure.add_axes([0.80, 0.37, 0.08, 0.05]), "리셋")
        self.random_button = Button(self.figure.add_axes([0.90, 0.37, 0.08, 0.05]), "랜덤 배치")
        self.truth_button = CheckButtons(self.figure.add_axes([0.80, 0.29, 0.18, 0.06]), ["Robot Perception"], [False])
        self.environment_button = RadioButtons(self.figure.add_axes([0.80, 0.025, 0.18, 0.15]), list(ENVIRONMENT_PRESETS), active=0)
        self.mode_button = RadioButtons(self.figure.add_axes([0.80, 0.19, 0.18, 0.09]), ["Manual", "CNN Agent"], active=0)
        self.start_button.on_clicked(self.start)
        self.stop_button.on_clicked(self.stop)
        self.reset_button.on_clicked(self.reset)
        self.random_button.on_clicked(self.randomize)
        self.truth_button.on_clicked(self.toggle_perception)
        self.environment_button.on_clicked(self.change_environment)
        self.mode_button.on_clicked(self.change_mode)
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
        self.board = Board(self.seed, self.environment, mi_parameters=self.mi_parameters)
        self.running = False
        self._reset_models()
        self.redraw()

    def randomize(self, event=None) -> None:
        self.seed += 1
        self.board = Board(self.seed, self.environment, mi_parameters=self.mi_parameters)
        self.running = False
        self._reset_models()
        self.redraw()

    def toggle_perception(self, label=None) -> None:
        self.robot_perception = not self.robot_perception
        self.redraw()

    def change_environment(self, label: str) -> None:
        self.environment = label
        self.board = Board(self.seed, self.environment, mi_parameters=self.mi_parameters)
        self.running = False
        self._reset_models()
        self.redraw()

    def open_parameters(self, event=None) -> None:
        self.running = False
        if self.parameter_window is not None:
            plt.close(self.parameter_window.figure)
        self.parameter_window = MIParameterWindow(self.board.sensor.field_model.profile, self.apply_parameters, lambda: self.board.sensor.field_model.profile)
        self.parameter_window.show()
        self.redraw()

    def open_relay_parameters(self, event=None) -> None:
        self.running = False
        if self.relay_window is not None:
            plt.close(self.relay_window.figure)
        self.relay_window = RelayParameterWindow(lambda: self.board.sensor.field_model.profile, self.apply_parameters)
        self.relay_window.show()
        self.redraw()

    def apply_parameters(self, parameters: dict) -> None:
        board = Board(self.seed, self.environment, mi_parameters=parameters)
        self.mi_parameters = parameters
        self.board = board
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

    def tick(self) -> None:
        if self.running:
            for _ in range(2):
                if self.detector is not None and self.detector_result.timestamp != self.board.time:
                    self.detector_result = self.detector.process_board(self.board)
                    rescue_event = self.rescue_manager.process(self.board, self.detector_result)
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
                if self.rescue_manager.all_rescued(self.board):
                    self.running = False
                    self.last_action = "stop"
                    break
            self.redraw()

    def redraw(self) -> None:
        self.map_axis.clear()
        draw_board(self.map_axis, self.board, self.robot_perception)
        if self.robot_perception:
            draw_patrol_overlay(self.map_axis, self.coverage, self.board.robot.position)
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
            f"Bx: {measurement.bx:.3f}\n"
            f"By: {measurement.by:.3f}\n"
            f"Bz: {measurement.bz:.3f}\n"
            f"|B|: {measurement.magnitude:.3f}\n"
            f"Nearest metal candidate: {nearest_candidate}\n"
            f"Mode: {self.navigation_mode}\n"
            f"Action: {self.last_action}\n"
            f"Detector: {self.detector_result.status}\n"
            f"Person probability: {self.detector_result.person_probability * 100:.1f}%\n"
            f"Prediction: {'미검증' if self.detector_result.status == 'UNCALIBRATED' else ('PERSON' if self.detector_result.predicted_label else 'NO PERSON')}\n"
            f"Ping decision: {self.detector_result.decision_mode.replace('_', ' ')}\n"
            f"Observed coverage: {self.coverage_ratio * 100:.1f}%"
        )
        if not self.robot_perception:
            text += f"\n구조 인원 (평가): {len(self.rescue_manager.rescued_indices)}"
        diagnostics = getattr(self.board.sensor, "mi_diagnostics", None)
        if diagnostics:
            calibration = diagnostics["calibration"]
            values = " / ".join(f"{value * 1000:.2f}" for value in diagnostics["output_rms_v"])
            text += f"\nMI: {SOURCE_LABELS[calibration['source']]}"
            relay = self.board.sensor.field_model.relay_summary
            if self.board.sensor.field_model.uses_physical_mi:
                text += f"\n전류: {calibration['tx_current_rms_a'] * 1000:g} mA / 코일 중계: {relay['gain']:.3g}배"
                text += f"\n수신 mV RMS: {values}\n실험 모델: CNN 미보정 · 구조 핑 비활성"
            else:
                scale = self.board.sensor.field_model.scaling
                text += f"\nV2 신호 · 반응 {scale['strength_factor']:.3g}배 / 범위 {self.board.sensor_range:.2f}"
            text += "\nBx·By·Bz: 정규화값 / CNN: V2 가중치"
        if self.rescue_manager.all_rescued(self.board):
            text += "\nSearch state: COMPLETE"
        elif self.coverage is not None and self.coverage.patrol_mode:
            text += "\nSearch state: PATROLLING"
            if self.coverage.patrol_active:
                patrol = self.coverage.patrol_scheduler.status()
                text += f"\nPatrol round: {patrol['round']} ({patrol['visited']}/{patrol['regions']} regions)"
        else:
            text += "\nSearch state: EXPLORING"
        if self.coverage is not None and self.coverage.relocating:
            text += "\n다른 구역으로 이동 중"
        if self.detector_error:
            text += f"\nDetector error: {self.detector_error}"
        if self.coverage_error:
            text += f"\nCoverage error: {self.coverage_error}"
        self.info_axis.text(0.0, 1.0, text, va="top", fontsize=9)
        self.figure.canvas.draw_idle()

    def show(self) -> None:
        plt.show()


if __name__ == "__main__":
    DesktopSimulationApp().show()
