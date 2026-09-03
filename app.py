import time

import matplotlib.pyplot as plt
import streamlit as st

from config import DEFAULT_SEED, ENVIRONMENT_PRESETS, SENSOR_RANGE, TIME_STEP
from evaluation.localization_score import evaluate_localization
from integration.controlled_stepper import ControlledBoardStepper
from integration.coverage_board_runtime import CoverageBoardRuntime
from integration.human_detector_runtime import HumanDetectorRuntime, HumanDetectorResult
from integration.rescue_manager import RescueManager
from simulation.board import Board
from ui.renderer import render_board, render_sensor_history


st.set_page_config(page_title="Human Detection Robot Simulation", layout="wide")


def initialize() -> None:
    if "seed" not in st.session_state:
        st.session_state.seed = DEFAULT_SEED
    if "environment" not in st.session_state:
        st.session_state.environment = next(iter(ENVIRONMENT_PRESETS))
    if "running" not in st.session_state:
        st.session_state.running = False
    if "noise_level" not in st.session_state:
        st.session_state.noise_level = 0.0
    if "board" not in st.session_state:
        st.session_state.board = Board(st.session_state.seed, st.session_state.environment, st.session_state.noise_level)
    if "navigation_mode" not in st.session_state:
        st.session_state.navigation_mode = "Manual"
    if "controlled_stepper" not in st.session_state:
        st.session_state.controlled_stepper = ControlledBoardStepper()
    if "rescue_manager" not in st.session_state:
        st.session_state.rescue_manager = RescueManager()
    if "detector_runtime" not in st.session_state:
        try:
            st.session_state.detector_runtime = HumanDetectorRuntime()
            st.session_state.detector_error = None
        except Exception as error:
            st.session_state.detector_runtime = None
            st.session_state.detector_error = str(error)
    if "coverage_runtime" not in st.session_state:
        try:
            st.session_state.coverage_runtime = CoverageBoardRuntime()
            st.session_state.coverage_error = None
        except Exception as error:
            st.session_state.coverage_runtime = None
            st.session_state.coverage_error = str(error)
    if "detector_result" not in st.session_state:
        st.session_state.detector_result = HumanDetectorResult(0.0, 0, "WARMING UP", 0.0, None, 0, 0)
    if "detector_processed_time" not in st.session_state:
        st.session_state.detector_processed_time = None


def reset(seed: int, environment: str) -> None:
    st.session_state.seed = seed
    st.session_state.environment = environment
    st.session_state.board = Board(seed, environment, st.session_state.noise_level)
    st.session_state.running = False
    st.session_state.controlled_stepper.reset()
    st.session_state.rescue_manager.reset()
    if st.session_state.detector_runtime is not None:
        st.session_state.detector_runtime.reset()
    if st.session_state.coverage_runtime is not None:
        st.session_state.coverage_runtime.reset()
    st.session_state.detector_result = HumanDetectorResult(0.0, 0, "WARMING UP", 0.0, None, 0, 0)
    st.session_state.detector_processed_time = None


def update_detector(board: Board) -> None:
    runtime = st.session_state.detector_runtime
    if runtime is None or st.session_state.detector_processed_time == board.time:
        return
    result = runtime.process_board(board)
    st.session_state.detector_result = result
    st.session_state.detector_processed_time = board.time
    rescue_event = st.session_state.rescue_manager.process(board, result)
    if rescue_event is not None:
        runtime.ignore_position(rescue_event.position)


def main() -> None:
    initialize()
    st.title("인명 탐지 로봇 시뮬레이션")
    with st.sidebar:
        st.header("제어")
        names = list(ENVIRONMENT_PRESETS)
        selected_environment = st.selectbox("환경", names, index=names.index(st.session_state.environment))
        robot_perception = st.toggle("로봇 인식 화면", value=False)
        st.session_state.navigation_mode = st.radio("Navigation Mode", ("Manual", "CNN Agent"), index=(0 if st.session_state.navigation_mode == "Manual" else 1))
        speed = st.slider("실행 속도", 1, 10, 4)
        noise_level = st.slider("환경 노이즈", 0.0, 5.0, float(st.session_state.noise_level), 0.1)
        if noise_level != st.session_state.noise_level:
            st.session_state.noise_level = noise_level
            st.session_state.board.set_enhanced_noise_level(noise_level)
        controls = st.columns(2)
        if controls[0].button("실행", width="stretch"):
            st.session_state.running = True
        if controls[1].button("정지", width="stretch"):
            st.session_state.running = False
        if st.button("초기화", width="stretch"):
            reset(st.session_state.seed, selected_environment)
            st.rerun()
        if st.button("랜덤 배치", width="stretch"):
            reset(st.session_state.seed + 1, selected_environment)
            st.rerun()
        if selected_environment != st.session_state.environment:
            reset(st.session_state.seed, selected_environment)
            st.rerun()
    board = st.session_state.board
    settings = board.settings
    update_detector(board)
    if st.session_state.rescue_manager.all_rescued(board):
        st.session_state.running = False
        st.session_state.last_action = "stop"
    status = "완료" if st.session_state.rescue_manager.all_rescued(board) else ("실행 중" if st.session_state.running else "정지")
    detector_result = st.session_state.detector_result
    metrics = st.columns(6)
    metrics[0].metric("상태", status)
    metrics[1].metric("시간", f"{board.time:.1f} s")
    metrics[2].metric("실제 사람", "숨김" if robot_perception else settings["people"])
    metrics[3].metric("금속", len(board.perception.metal_candidates()) if robot_perception else settings["metals"])
    metrics[4].metric("장애물", settings["obstacles"])
    metrics[5].metric("로봇 사람 추정", len(board.detection_tracker.markers))
    st.caption(f"구조 처리된 사람: {len(st.session_state.rescue_manager.rescued_indices)}명 · 환경 노이즈: {st.session_state.noise_level:.1f}")
    map_column, state_column = st.columns([3, 1])
    with map_column:
        figure = render_board(board, robot_perception)
        st.pyplot(figure, width="stretch")
        plt.close(figure)
    with state_column:
        measurement = board.measurements[-1]
        st.subheader("자기장 센서")
        st.metric("|B|", f"{measurement.magnitude:.3f}")
        st.write(f"Bx: {measurement.bx:.3f}")
        st.write(f"By: {measurement.by:.3f}")
        st.write(f"Bz: {measurement.bz:.3f}")
        st.write(f"측정 범위: {SENSOR_RANGE:.1f}")
        st.write(f"금속 후보: {len(board.perception.metal_candidates())}")
        distances = board.perception.candidate_distances(board.robot.position)
        st.write(f"가장 가까운 금속 후보: {min(distances):.2f}" if len(distances) else "가장 가까운 금속 후보: 없음")
        st.write(f"인식 장애물 점: {len(board.perception.obstacle_points)}")
        st.write(f"사람 추정 마커: {len(board.detection_tracker.markers)}")
        if not robot_perception:
            evaluation = evaluate_localization(board.detection_tracker.markers, board.people)
            st.subheader("사람 위치 평가")
            st.write(f"위치 점수: {evaluation.localization_score:.1f} / 100")
            st.write(f"매칭: {len(evaluation.matches)}")
            st.write(f"중복·오탐: {evaluation.false_positives}")
            st.write(f"미탐: {evaluation.missed_people}")
            if evaluation.mean_distance is not None:
                st.write(f"평균 거리 오차: {evaluation.mean_distance:.2f}")
        st.subheader("로봇")
        st.write(f"X: {board.robot.position[0]:.2f}")
        st.write(f"Y: {board.robot.position[1]:.2f}")
        st.write(f"방향: {board.robot.heading:.3f}")
        st.write(f"동작: {st.session_state.get('last_action', 'stop')}")
        st.subheader("AI Detector")
        st.metric("Person probability", f"{detector_result.person_probability * 100:.1f}%")
        st.metric("Prediction", "PERSON" if detector_result.predicted_label else "NO PERSON")
        st.metric("Detector status", detector_result.status)
        st.write(f"Window: {detector_result.window_size}/16")
        if st.session_state.detector_error:
            st.error(st.session_state.detector_error)
        st.subheader("Coverage Agent")
        st.write(f"Mode: {st.session_state.navigation_mode}")
        st.write(f"Action: {st.session_state.get('last_action', 'stop')}")
        if st.session_state.rescue_manager.all_rescued(board):
            st.write("Search state: COMPLETE")
        elif st.session_state.coverage_runtime is not None and st.session_state.coverage_runtime.patrol_mode:
            st.write("Search state: PATROLLING")
        else:
            st.write("Search state: EXPLORING")
        st.write(f"Observed coverage: {st.session_state.get('coverage_ratio', 0.0) * 100:.1f}%")
        if st.session_state.coverage_error:
            st.error(st.session_state.coverage_error)
    history_figure = render_sensor_history(board)
    st.pyplot(history_figure, width="stretch")
    plt.close(history_figure)
    if st.session_state.running:
        for _ in range(speed):
            if st.session_state.rescue_manager.all_rescued(board):
                st.session_state.running = False
                st.session_state.last_action = "stop"
                break
            if st.session_state.navigation_mode == "CNN Agent" and st.session_state.coverage_runtime is not None:
                action, coverage = st.session_state.coverage_runtime.decide(board)
                st.session_state.last_action = action
                st.session_state.coverage_ratio = coverage
                st.session_state.controlled_stepper.step(board, action, 0.4)
            else:
                st.session_state.last_action = "forward"
                st.session_state.coverage_ratio = board.perception.observed.mean()
                board.step(TIME_STEP)
            update_detector(board)
        time.sleep(0.06)
        st.rerun()


if __name__ == "__main__":
    main()

