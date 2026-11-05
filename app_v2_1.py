import time
from io import BytesIO

import matplotlib.pyplot as plt
import streamlit as st

from config_v2_1 import DEFAULT_SEED, ENVIRONMENT_PRESETS, SENSOR_RANGE, TIME_STEP
from evaluation.localization_score_v2_1 import evaluate_localization
from integration.controlled_stepper_v2_1 import ControlledBoardStepper
from integration.coverage_board_runtime_v2_1 import CoverageBoardRuntime
from integration.human_detector_runtime_v2_1 import HumanDetectorRuntime, HumanDetectorResult
from integration.rescue_manager_v2_1 import RescueManager
from simulation.board_v2_1 import Board
from ui.renderer_v2_1 import render_board, render_sensor_history
from ui.patrol_overlay_v2_1 import draw_patrol_overlay
from ui.mi_parameters_v2_1 import SOURCE_LABELS
from simulation.mi_settings_v2_1 import DESIGN_PARAMETERS, default_settings, design_values, design_profile, expected_voltage_mv, make_settings, settings_rows
from simulation.relay_settings_v2_1 import relay_values, relay_summary, with_relay


st.set_page_config(page_title="Human Detection Robot Simulation", layout="wide")


def show_figure(figure) -> None:
    buffer = BytesIO()
    figure.savefig(buffer, format="png", dpi=85, bbox_inches="tight")
    st.image(buffer.getvalue(), width="stretch")


def initialize() -> None:
    if "mi_parameters" not in st.session_state:
        st.session_state.mi_parameters = None
    if "seed" not in st.session_state:
        st.session_state.seed = DEFAULT_SEED
    if "environment" not in st.session_state:
        st.session_state.environment = next(iter(ENVIRONMENT_PRESETS))
    if "running" not in st.session_state:
        st.session_state.running = False
    if "board" not in st.session_state:
        st.session_state.board = Board(st.session_state.seed, st.session_state.environment, mi_parameters=st.session_state.mi_parameters)
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


def reset(seed: int, environment: str, parameters: dict | None = None) -> None:
    active_parameters = st.session_state.mi_parameters if parameters is None else parameters
    board = Board(seed, environment, mi_parameters=active_parameters)
    st.session_state.mi_parameters = active_parameters
    st.session_state.seed = seed
    st.session_state.environment = environment
    st.session_state.board = board
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


def parameter_controls() -> None:
    profile = st.session_state.board.sensor.field_model.profile
    revision = st.session_state.get("mi_settings_revision", 0)
    initial_mode = "실험값" if profile.get("input_mode") == "measured" else "예상값"
    with st.expander("자기장 설정", expanded=False):
        source = st.session_state.board.sensor.field_model.calibration.source
        st.caption("현재 적용: " + ("실험값" if source == "measured_transfer" else "예상값"))
        scaling = st.session_state.board.sensor.field_model.scaling
        st.caption(f"현재 적용: 반응 {scaling['strength_factor']:.4g}배 · 센서 범위 {scaling['sensor_range']:.2f}")
        st.caption("V2의 신호 형태·잡음·CNN·핑 판단을 유지하고 반응 세기와 범위만 조절합니다.")
        mode = st.radio("사용할 값", ("예상값", "실험값"), index=int(initial_mode == "실험값"),
                        key=f"mi_simple_mode_{revision}")
        location = "coil_input"
        st.caption("결과는 수신 코일 전압입니다. 코일 중계는 별도 설정에서 정합니다.")
        st.markdown("**① 실험 설계·조건**")
        st.caption("예상값은 아래 결과를 자동 계산합니다. 실험값은 같은 조건에서 측정한 결과를 넣으세요.")
        initial_values = design_values(profile)
        values = {}
        for key, label, _ in DESIGN_PARAMETERS:
            integer = key.endswith("_turns")
            values[key] = st.number_input(label, min_value=1 if integer else (0.0 if key == "receiver_output_noise_rms_v" else 0.000001),
                                         value=int(initial_values[key]) if integer else float(initial_values[key]),
                                         step=1 if integer else 0.1, key=f"mi_simple_design_{key}_{revision}")
        layout_key = f"mi_simple_layout_{revision}"
        if layout_key not in st.session_state:
            st.session_state[layout_key] = settings_rows(profile)
        st.markdown("**② 예상 결과 — 자동 계산**" if mode == "예상값" else "**② 실험 결과 — 직접 입력**")
        st.caption("중계 없음·정렬 0도 기준. 실험값은 서로 다른 거리 두 곳 이상 입력하고 미측정 행은 비워 두세요.")
        distances, voltages = [], []
        preview = None
        try:
            preview = design_profile(profile, values)
        except (ValueError, KeyError) as error:
            st.error(str(error))
        for i, (distance, voltage) in enumerate(st.session_state[layout_key]):
            columns = st.columns(2)
            d = columns[0].number_input(f"거리 {i + 1} (cm)", min_value=0.001, value=float(distance),
                                       key=f"mi_simple_distance_{revision}_{i}")
            distances.append(d)
            if mode == "예상값":
                value = expected_voltage_mv(preview, d, location) if preview else 0.0
                columns[1].text_input(f"예상 전압 {i + 1} (mV RMS)", value=f"{value:.6g}", disabled=True,
                                     key=f"mi_simple_prediction_{revision}_{i}_{value:.9g}")
                voltages.append(None)
            else:
                same_location = profile.get("voltage_location") == location
                value = float(voltage) if initial_mode == "실험값" and same_location and voltage is not None else None
                voltages.append(columns[1].number_input(f"실험 전압 {i + 1} (mV RMS)", min_value=0.000001, value=value,
                                                        key=f"mi_simple_measured_{revision}_{location}_{i}"))
        st.caption("입력·계산만으로는 반영되지 않습니다. 아래 적용 버튼을 누르세요.")
        buttons = st.columns(2)
        apply_clicked = buttons[0].button("적용", key="mi_simple_apply")
        reset_clicked = buttons[1].button("예상값으로 처음부터 초기화", key="mi_simple_reset")
        if apply_clicked or reset_clicked:
            try:
                parameters = default_settings() if reset_clicked else make_settings(profile, values, mode, location, distances, voltages)
                reset(st.session_state.seed, st.session_state.environment, parameters)
            except (ValueError, KeyError, OSError) as error:
                st.error(str(error))
            else:
                st.session_state.mi_settings_revision = revision + 1
                st.rerun()


def relay_controls() -> None:
    profile = st.session_state.board.sensor.field_model.profile
    revision = st.session_state.get("mi_settings_revision", 0)
    active = relay_values(profile)
    with st.expander("코일 중계 설정", expanded=False):
        st.caption("중계 전압의 크기 배율만 물체 반응과 센서 범위에 반영합니다. 위상은 표시만 합니다.")
        st.caption("같은 간격으로 정렬한 공진 중계 코일의 예상 회로값을 항상 계산합니다.")
        enabled = st.toggle("코일 중계 사용", value=active["enabled"], key=f"relay_enabled_{revision}")
        count = st.number_input("중계 장치 개수", min_value=1, max_value=8, value=active["count"], key=f"relay_count_{revision}")
        spacing = st.number_input("이웃한 코일 장치 중심 간격 (cm)", min_value=0.001, value=active["spacing_cm"], key=f"relay_spacing_{revision}")
        axis = st.selectbox("전달 방향과 정렬할 코일 축", ("X", "Y", "Z"),
                            index=("X", "Y", "Z").index(active["axis"]), key=f"relay_axis_{revision}")
        draft = None
        try:
            draft = with_relay(profile, {"enabled": enabled, "count": count, "spacing_cm": spacing, "axis": axis})
            summary = relay_summary(draft)
            st.write(f"송신–수신 전체 거리: {summary['distance_m'] * 100:g} cm")
            st.write(f"중계 없음: {summary['direct_rms_v'] * 1000:.6g} mV RMS")
            st.write(f"현재 설정: {summary['input_rms_v'] * 1000:.6g} mV RMS")
            st.metric("수신 전압 배율", f"{summary['gain']:.4g}배")
            st.caption(f"위상 변화: {summary['phase_deg']:.1f}°")
        except (ValueError, KeyError, TypeError) as error:
            st.error(str(error))
        st.caption("3축 장치의 선택 축 코일 쌍을 사용합니다. 실측 전 L/R/C 가정값이며 신호가 감소할 수도 있습니다.")
        st.caption("동축 링크 계산값을 신호 세기 배율로 요약하는 합성 근사이며 실제 지도에 코일을 배치하지 않습니다.")
        if st.button("적용", key="relay_apply", disabled=draft is None):
            reset(st.session_state.seed, st.session_state.environment, draft)
            st.session_state.mi_settings_revision = revision + 1
            st.rerun()


def main() -> None:
    initialize()
    st.title("인명 탐지 로봇 시뮬레이션 V2.1")
    with st.sidebar:
        st.header("제어")
        names = list(ENVIRONMENT_PRESETS)
        selected_environment = st.selectbox("환경", names, index=names.index(st.session_state.environment))
        robot_perception = st.toggle("로봇 인식 화면", value=False)
        st.session_state.navigation_mode = st.radio("Navigation Mode", ("Manual", "CNN Agent"), index=(0 if st.session_state.navigation_mode == "Manual" else 1))
        speed = st.slider("실행 속도", 1, 10, 4)
        parameter_controls()
        relay_controls()
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
    if not robot_perception:
        st.caption(f"구조 처리된 사람 (평가): {len(st.session_state.rescue_manager.rescued_indices)}명")
    diagnostics = getattr(board.sensor, "mi_diagnostics", None)
    if diagnostics:
        st.info("MI 데이터: " + SOURCE_LABELS[diagnostics["calibration"]["source"]] + " · Bx/By/Bz는 정규화 입력값입니다.")
        bounds = diagnostics["calibration"]["measured_range_m"]
        if bounds:
            st.caption(f"실험값 적용 범위: {bounds[0] * 100:g}~{bounds[1] * 100:g} cm. 범위 밖은 기본 이론 계산을 사용합니다.")
        st.caption("기존 V2 CNN을 재사용 중입니다. V2.1 신호에 대한 재학습·정확도 검증 전입니다.")
    map_column, state_column = st.columns([3, 1])
    with map_column:
        figure = render_board(board, robot_perception)
        if robot_perception:
            draw_patrol_overlay(figure.axes[0], st.session_state.coverage_runtime, board.robot.position)
        show_figure(figure)
        plt.close(figure)
    with state_column:
        measurement = board.measurements[-1]
        st.subheader("자기장 센서")
        if diagnostics:
            st.write("수신 출력 (mV RMS): " + " / ".join(f"{v * 1000:.3f}" for v in diagnostics["output_rms_v"]))
            if diagnostics["clipped"]:
                st.warning("수신 출력이 설정된 최대값에서 클리핑되었습니다.")
        st.metric("|B|", f"{measurement.magnitude:.3f}")
        st.write(f"Bx: {measurement.bx:.3f}")
        st.write(f"By: {measurement.by:.3f}")
        st.write(f"Bz: {measurement.bz:.3f}")
        st.write(f"측정 범위: {board.sensor_range:.1f}")
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
        if detector_result.status == "UNCALIBRATED":
            st.warning("실험 전달 모델의 CNN은 미보정입니다. 아래 확률은 참고값이며 구조 핑을 생성하지 않습니다.")
        st.metric("Person probability", f"{detector_result.person_probability * 100:.1f}%")
        st.metric("Prediction", "미검증" if detector_result.status == "UNCALIBRATED" else ("PERSON" if detector_result.predicted_label else "NO PERSON"))
        st.metric("Detector status", detector_result.status)
        st.metric("Ping decision", detector_result.decision_mode.replace("_", " "))
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
            if st.session_state.coverage_runtime.patrol_active:
                patrol = st.session_state.coverage_runtime.patrol_scheduler.status()
                st.write(f"Patrol round: {patrol['round']} ({patrol['visited']}/{patrol['regions']} regions)")
        else:
            st.write("Search state: EXPLORING")
        if st.session_state.coverage_runtime is not None and st.session_state.coverage_runtime.relocating:
            st.write("다른 구역으로 이동 중")
        st.write(f"Observed coverage: {st.session_state.get('coverage_ratio', 0.0) * 100:.1f}%")
        if st.session_state.coverage_error:
            st.error(st.session_state.coverage_error)
    history_figure = render_sensor_history(board)
    show_figure(history_figure)
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
