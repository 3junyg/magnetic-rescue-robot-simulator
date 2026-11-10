from _bootstrap_v2 import prepare

prepare()

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.widgets import Button, RadioButtons, Slider
import numpy as np

from evaluation.narrow_scenes_v2 import create_narrow_scene
from integration.controlled_stepper_v2 import ControlledBoardStepper
from integration.coverage_board_runtime_v2 import CoverageBoardRuntime
from ui.renderer_v2 import draw_board


class NarrowNavigationApp:
    def __init__(self) -> None:
        self.seed = 11
        self.width = 2.8
        self.kind = 'bend'
        self.step_limit = 300
        self.running = False
        self.agent = CoverageBoardRuntime()
        self.stepper = ControlledBoardStepper()
        self.figure = plt.figure(figsize=(14, 8))
        self.figure.canvas.manager.set_window_title('V2 Narrow Passage Test')
        self.actual_axis = self.figure.add_axes([0.04, 0.29, 0.44, 0.63])
        self.perception_axis = self.figure.add_axes([0.52, 0.29, 0.44, 0.63])
        self.info = self.figure.text(0.04, 0.22, '', fontsize=11, va='top')
        self.buttons = []
        for x, title, callback in ((0.04, 'Start', self.start), (0.17, 'Stop', self.stop), (0.30, 'Reset', self.reset), (0.43, 'Step', self.single_step), (0.56, 'New Seed', self.new_seed)):
            button = Button(self.figure.add_axes([x, 0.025, 0.11, 0.055]), title)
            button.on_clicked(callback)
            self.buttons.append(button)
        self.width_slider = Slider(self.figure.add_axes([0.13, 0.11, 0.36, 0.025]), 'Width', 2.0, 6.0, valinit=self.width, valstep=0.1)
        self.width_slider.on_changed(self.change_width)
        self.speed_slider = Slider(self.figure.add_axes([0.59, 0.11, 0.13, 0.025]), 'Speed', 1, 8, valinit=2, valstep=1)
        self.kind_radio = RadioButtons(self.figure.add_axes([0.77, 0.025, 0.19, 0.18]), ('L bend', 'Straight', 'Corner', 'Room + gap', 'Wide / narrow'), active=0)
        self.kind_radio.on_clicked(self.change_kind)
        self.help_figure = None
        self.help_button = Button(self.figure.add_axes([0.78, 0.22, 0.18, 0.045]), '화면 설명')
        self.help_button.on_clicked(self.show_help)
        self.timer = self.figure.canvas.new_timer(interval=100)
        self.timer.add_callback(self.tick)
        self.reset()
        self.timer.start()
        self.figure.canvas.mpl_connect('close_event', lambda event: self.timer.stop())

    def reset(self, event=None) -> None:
        self.running = False
        self.board = create_narrow_scene(self.seed, self.width, self.kind)
        self.agent.reset()
        self.stepper.reset()
        self.steps = self.collisions = self.blocked = 0
        self.reached = False
        self.action = 'stop'
        self.coverage = float(self.board.perception.observed.mean())
        self.path = [self.board.robot.position.copy()]
        self.redraw()

    def start(self, event=None) -> None:
        self.running = not self.reached and self.steps < self.step_limit

    def stop(self, event=None) -> None:
        self.running = False
        self.redraw()

    def change_width(self, value: float) -> None:
        self.width = float(value)
        self.reset()

    def change_kind(self, value: str) -> None:
        self.kind = {'L bend': 'bend', 'Straight': 'straight', 'Corner': 'corner', 'Room + gap': 'room', 'Wide / narrow': 'choice'}[value]
        self.reset()

    def new_seed(self, event=None) -> None:
        self.seed += 1
        self.reset()

    def advance(self) -> None:
        if self.reached or self.steps >= self.step_limit:
            return
        self.action, self.coverage = self.agent.decide(self.board)
        self.stepper.step(self.board, self.action, 0.4)
        self.steps += 1
        self.collisions += int(not self.board._is_free(self.board.robot.position, self.board.robot.radius))
        self.blocked += int(self.action == 'forward' and not self.board.robot.moving)
        self.path.append(self.board.robot.position.copy())
        self.reached = bool(self.board.robot.position[1] > 62 if self.kind == 'bend' else self.board.robot.position[0] > 64)
        if self.reached or self.collisions or self.steps >= self.step_limit:
            self.running = False

    def single_step(self, event=None) -> None:
        self.running = False
        self.advance()
        self.redraw()

    def tick(self) -> None:
        if self.running:
            for _ in range(int(self.speed_slider.val)):
                self.advance()
                if not self.running:
                    break
            self.redraw()

    def redraw(self) -> None:
        for axis, perception in ((self.actual_axis, False), (self.perception_axis, True)):
            axis.clear()
            draw_board(axis, self.board, perception)
            axis.set_title('로봇 센서 인식 화면' if perception else '실제 환경 (평가용)')
            if self.kind == 'bend':
                axis.set_xlim(43, 66)
                axis.set_ylim(35, 68)
                axis.axhline(62, color='green', linestyle='--', label='목표선')
            elif self.kind in ('room', 'choice'):
                axis.set_xlim(10, 95)
                axis.set_ylim(0, 80)
                axis.axvline(64, color='green', linestyle='--', label='목표선')
            else:
                axis.set_xlim(48, 70)
                axis.set_ylim(30, 51)
                axis.axvline(64, color='green', linestyle='--', label='목표선')
            path = np.asarray(self.path)
            axis.plot(path[:, 0], path[:, 1], color='#42a5f5', linewidth=1.5)
            axis.add_patch(Circle(self.board.robot.position, self.board.robot.radius, color='#1565c0', alpha=0.3, label='로봇 몸 크기'))
            handles, labels = axis.get_legend_handles_labels()
            names = {'Actual metal': '실제 금속', 'Robot': '로봇 중심', 'Perceived obstacle': '관측한 장애물', 'Metal candidate': '금속 추정 후보'}
            axis.legend(handles, [names.get(label, label) for label in labels], loc='upper right', fontsize=8)
        planner = self.agent.active_planner
        if planner.last_waypoint is not None:
            point = planner.last_waypoint
            self.perception_axis.scatter(*point, marker='x', color='green', s=60, label='다음 이동 목표')
            handles, labels = self.perception_axis.get_legend_handles_labels()
            self.perception_axis.legend(handles, [names.get(label, label) for label in labels], loc='upper right', fontsize=8)
        status = 'PASSED' if self.reached else ('STEP LIMIT' if self.steps >= self.step_limit else ('RUNNING' if self.running else 'STOPPED'))
        self.info.set_text(f'{status} | seed={self.seed} | width={self.width:.1f} | robot diameter={2 * self.board.robot.radius:.1f}\n'
                           f'step={self.steps} | action={self.action} | control={self.agent.control_mode} | wall penetration={self.collisions} | blocked forward={self.blocked}\n'
                           '이동 전용 시험: 입구 쪽 지도는 미리 관측한 상태입니다. 초록 점선 = 목표선 | 오른쪽 버튼: 화면 설명')
        self.figure.canvas.draw_idle()

    def show_help(self, event=None) -> None:
        if self.help_figure is not None and plt.fignum_exists(self.help_figure.number):
            self.help_figure.show()
            return
        self.help_figure = plt.figure(figsize=(12, 8))
        self.help_figure.canvas.manager.set_window_title('좁은 통로 시험 — 화면 설명')
        self.help_figure.text(0.05, 0.94, '좁은 통로 시각화 안내', fontsize=19, weight='bold')
        left = (
            '왼쪽: 실제 환경 (평가용)\n\n'
            '진한 회색: 실제 벽·장애물\n'
            '밝은 하늘색: 통로와 빈 공간\n'
            '갈색 네모: 실제 금속 물체\n\n'
            '두 화면의 공통 표시\n\n'
            '파란 점: 로봇 중심\n'
            '작은 반투명 원: 로봇 몸 크기\n'
            '파란 화살표: 로봇 방향\n'
            '큰 파란 점선 원: 센서 범위\n'
            '하늘색 선: 이동 경로\n'
            '초록 점선: 목표선 (중심이 넘으면 통과)\n\n'
            '오른쪽: 로봇 센서 인식 화면\n\n'
            '파란 음영: 관측한 영역\n'
            '노랑·갈색 음영: 자기장 이상 세기\n'
            '  진할수록 강함. 사람을 뜻하지 않음.\n'
            '검은 작은 점: 관측한 장애물 지점\n'
            '주황 빈 네모: 금속 추정 후보\n'
            '초록 ×: 다음 이동 목표'
        )
        right = (
            '상태와 수치\n\n'
            'RUNNING / STOPPED: 실행 중 / 정지\n'
            'PASSED: 목표선 통과\n'
            'STEP LIMIT: 300 timestep 내 통과 못함\n'
            'seed: 초기 조건 재현 번호\n'
            'width: 통로 폭\n'
            'robot diameter: 로봇 지름 (2.4)\n'
            'step: 진행한 timestep 수\n'
            'action: 현재 이동 행동\n'
            'control: NORMAL 일반 / PRECISE 정밀 / RECOVERY 탈출\n'
            'wall penetration: 로봇 몸의 벽 침범 횟수\n'
            'blocked forward: 전진했지만 이동 못한 횟수\n'
            '  이동 차단이지, 벽 침범이라는 뜻은 아님.\n\n'
            '조작\n\n'
            'Start / Stop: 실행 / 정지\n'
            'Reset: 같은 조건으로 초기화\n'
            'Step: 한 timestep만 진행\n'
            'New Seed: seed 증가 후 초기화\n'
            'Width: 폭 변경 후 초기화\n'
            'Speed: 화면 갱신당 진행할 timestep 수\n'
            'L bend / Straight / Corner: 통로 유형\n\n'
            'Room + gap: 넓은 방 사이의 좁은 문\n'
            'Wide / narrow: 넓은 문과 좁은 문을 선택\n\n'
            '이 앱은 이동 전용 시험입니다. 사람은 없습니다.\n'
            '입구 쪽 지도는 미리 관측한 상태입니다.\n'
            '실제 환경 화면은 Agent 입력이 아닙니다.'
        )
        self.help_figure.text(0.05, 0.87, left, fontsize=11, va='top', linespacing=1.35)
        self.help_figure.text(0.53, 0.87, right, fontsize=11, va='top', linespacing=1.35)
        self.help_figure.show()

    def show(self) -> None:
        plt.show()


if __name__ == '__main__':
    NarrowNavigationApp().show()
