from __future__ import annotations

from copy import deepcopy
from collections.abc import Callable
import math

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons, TextBox

from simulation.mi_calibration_v2_1 import MICalibration, DEFAULT_PROFILE
from simulation.mi_settings_v2_1 import DESIGN_PARAMETERS, default_settings, design_values, design_profile, expected_voltage_mv, make_settings, settings_rows
from simulation.signal_scaling_v2_1 import signal_scaling


DISTANCES_M = (0.2, 0.3, 0.4, 0.5, 0.6, 0.8)
SOURCE_LABELS = {
    "provisional_theory": "계산식으로 만든 예상값",
    "manual_assumption": "직접 입력한 예상값",
    "measured_transfer": "실험값 적용 중",
}
PARAMETERS = (
    ("frequency_hz", "자기장을 반복하는 횟수 (Hz)", 1.0),
    ("tx_current_rms_a", "보내는 신호의 세기 — 전류 (mA)", 1000.0),
    ("receiver_gain", "받은 신호의 확대 배율 (배)", 1.0),
    ("receiver_output_noise_rms_v", "측정값의 흔들림 — 잡음 (mV)", 1000.0),
)
PARAMETER_HELP = {
    "frequency_hz": "송신 주파수입니다. 2000이면 자기장 파형이 1초에 2000번 반복됩니다. 모르면 2000을 유지하세요.",
    "tx_current_rms_a": "송신 코일에 흐르는 전류입니다. 크게 하면 보내는 자기장이 강해집니다. 계측기의 기본파 RMS 전류를 mA로 입력하세요.",
    "receiver_gain": "수신 회로가 받은 신호를 몇 배 키우는지 정합니다. 크게 해도 사람이 만드는 반응 자체가 강해지는 것은 아닙니다.",
    "receiver_output_noise_rms_v": "수신 회로 출력에 더하는 잡음의 크기입니다. 크게 하면 측정값이 더 불안정해집니다. 0이면 이 항목의 추가 잡음만 꺼집니다.",
}
SHORT_HELP = {
    "frequency_hz": "1초에 몇 번 반복할지 정합니다. 처음에는 2000을 유지하세요.",
    "tx_current_rms_a": "값을 높이면 송신 코일이 만드는 자기장이 강해집니다.",
    "receiver_gain": "수신 회로의 확대 배율입니다. 대상의 반응 세기와는 다릅니다.",
    "receiver_output_noise_rms_v": "값을 높이면 측정값이 더 흔들립니다. 0은 이 추가 잡음만 끕니다.",
}


class MITextBox(TextBox):
    def _resize(self, event) -> None:
        if self.capturekeystrokes:
            self.stop_typing()


def transfer_defaults(profile: dict) -> dict:
    manual = deepcopy(profile.get("manual_transfer", {}))
    if not manual.get("rows"):
        calibration = MICalibration(profile_overrides={**profile, "manual_transfer": {"enabled": False}})
        manual = {
            "enabled": False,
            "reference_current_rms_a": 0.05,
            "rows": [[d, calibration.direct_transfer(d) * 0.05] for d in DISTANCES_M],
        }
    if manual.get("provenance") == "measured":
        rows = {float(d): v for d, v in manual["rows"]}
        for distance in DISTANCES_M:
            if len(rows) >= 6:
                break
            rows.setdefault(distance, None)
        manual["rows"] = sorted(rows.items())
    return manual


def make_parameters(base: dict, values: dict, enabled: bool, reference_ma: float, voltages_mv: list[float | None],
                    measured: bool = False, voltage_location: str = "coil_input", distances_cm: list[float] | None = None) -> dict:
    profile = deepcopy(base)
    for key, label, scale in PARAMETERS:
        value = float(values[key])
        minimum = 0.0
        if not math.isfinite(value) or value < minimum or (value == 0 and key != "receiver_output_noise_rms_v"):
            raise ValueError(f"{label}: {'0 이상의' if key == 'receiver_output_noise_rms_v' else '0보다 큰'} 숫자를 입력하세요.")
        profile[key] = value / scale
    manual = transfer_defaults(base)
    if not enabled:
        retained = deepcopy(base.get("manual_transfer", manual))
        retained["enabled"] = False
        profile["manual_transfer"] = retained
        return MICalibration(profile_overrides=profile).profile
    if len(voltages_mv) != len(manual["rows"]):
        raise ValueError("거리별 전압의 개수가 맞지 않습니다.")
    distances = [row[0] * 100 for row in manual["rows"]] if distances_cm is None else distances_cm
    if len(distances) != len(voltages_mv):
        raise ValueError("거리와 전압의 행 수가 다릅니다.")
    rows = [[float(distance) / 100, float(value) / 1000]
            for distance, value in zip(distances, voltages_mv) if value is not None]
    profile["manual_transfer"] = {
        "enabled": bool(enabled),
        "reference_current_rms_a": float(reference_ma) / 1000.0,
        "rows": rows,
        "provenance": "measured" if measured else "assumed",
        "voltage_location": voltage_location,
        "measurement_gain": profile["receiver_gain"],
        "frequency_hz": profile["frequency_hz"],
        "relay_count": 0,
        "rx_angle_deg": 0.0,
        "medium": profile["measurement_medium"],
    }
    if measured:
        profile["input_mode"] = "measured"
    else:
        profile["input_mode"] = None
    return MICalibration(profile_overrides=profile).profile


class MIParameterWindow:
    def __init__(self, profile: dict, on_apply: Callable[[dict], None], get_profile: Callable[[], dict] | None = None) -> None:
        self.profile = deepcopy(profile)
        self.on_apply = on_apply
        self.get_profile = get_profile
        self.restoring = False
        self.mode = "실험값" if profile.get("manual_transfer", {}).get("enabled") and profile.get("manual_transfer", {}).get("provenance") == "measured" else "예상값"
        self.figure = plt.figure(figsize=(12, 9.5))
        self.figure.canvas.manager.set_window_title("V2.1 자기장 설정")
        self.figure.text(0.06, 0.96, "자기장 설정", fontsize=18, weight="bold")
        self.figure.text(0.06, 0.925, "예상값: 설계만 입력하면 결과를 계산합니다.  실험값: 설계와 측정 결과를 입력합니다.", fontsize=10)
        self.summary = self.figure.text(0.06, 0.89, "", fontsize=10, color="#175e83")
        self.mode_radio = RadioButtons(self.figure.add_axes([0.06, 0.79, 0.30, 0.08]), ["예상값", "실험값"], active=int(self.mode == "실험값"))
        self.figure.text(0.42, 0.82, "설정은 물체 반응 세기와 센서 범위만 조절합니다.\nV2의 신호 형태·잡음·CNN·핑 판단은 유지합니다.", fontsize=10)
        self.figure.text(0.06, 0.755, "① 실험 설계·조건", fontsize=12, weight="bold")
        self.boxes = {}
        values = design_values(profile)
        for i, (key, label, _) in enumerate(DESIGN_PARAMETERS):
            col, row = divmod(i, 4)
            x = 0.06 if col == 0 else 0.54
            bx = 0.35 if col == 0 else 0.84
            y = 0.69 - row * 0.06
            self.figure.text(x, y + 0.011, label, fontsize=10)
            self.boxes[key] = MITextBox(self.figure.add_axes([bx, y, 0.11, 0.037]), "", initial=f"{values[key]:.6g}")
        self.result_title = self.figure.text(0.06, 0.455, "", fontsize=12, weight="bold")
        self.figure.text(0.10, 0.42, "거리 (cm)", fontsize=10)
        self.figure.text(0.58, 0.42, "수신 전압 (mV RMS)", fontsize=10)
        rows = settings_rows(profile)
        self.distances, self.voltages = [], []
        for i, (distance, voltage) in enumerate(rows):
            y = 0.37 - i * min(0.043, 0.26 / len(rows))
            self.distances.append(MITextBox(self.figure.add_axes([0.10, y, 0.24, 0.03]), "", initial=f"{distance:.6g}"))
            self.figure.text(0.43, y + 0.006, "→", fontsize=12)
            self.voltages.append(MITextBox(self.figure.add_axes([0.58, y, 0.35, 0.03]), "", initial="" if voltage is None else f"{voltage:.6g}"))
        self.measured_cache = [box.text for box in self.voltages] if self.mode == "실험값" else [""] * len(rows)
        self.figure.text(0.06, 0.105, "중계 없음·정렬 0도 기준입니다. 실험은 두 거리 이상 입력하고, 미측정 행은 비워 두세요.", fontsize=9)
        self.reset_button = Button(self.figure.add_axes([0.06, 0.04, 0.43, 0.045]), "예상값으로 처음부터 초기화")
        self.apply_button = Button(self.figure.add_axes([0.55, 0.04, 0.39, 0.045]), "적용")
        self.message = self.figure.text(0.06, 0.012, "", fontsize=10, color="#175e83")
        self.reset_button.on_clicked(self.reset_to_defaults)
        self.apply_button.on_clicked(self.apply)
        self.mode_radio.on_clicked(self.change_mode)
        for box in [*self.boxes.values(), *self.distances, *self.voltages]:
            box.on_text_change(self.refresh_results)
        self.update_summary()
        self.refresh_results()
        self.message.set_text("설정 변경은 아래 ‘적용’을 눌러야 시뮬레이션에 반영됩니다.")

    def location(self) -> str:
        return "coil_input"

    def read_design(self) -> dict:
        return {key: float(box.text) for key, box in self.boxes.items()}

    def update_summary(self) -> None:
        source = MICalibration(profile_overrides=self.profile).source
        scale = signal_scaling(MICalibration(profile_overrides=self.profile))
        self.summary.set_text(f"현재 적용: 반응 {scale['strength_factor']:.4g}배 · 범위 {scale['sensor_range']:.2f}  |  {'실험값' if source == 'measured_transfer' else '예상값'}")

    def change_mode(self, label=None) -> None:
        if self.restoring:
            return
        if self.mode == "실험값":
            self.measured_cache = [box.text for box in self.voltages]
        self.mode = self.mode_radio.value_selected
        if self.mode == "실험값":
            for box, value in zip(self.voltages, self.measured_cache):
                box.text_disp.set_text(value)
        self.refresh_results()

    def refresh_results(self, event=None) -> None:
        if self.restoring:
            return
        measured = self.mode == "실험값"
        self.result_title.set_text("② 실험 결과 — 직접 입력" if measured else "② 예상 결과 — 설계 조건으로 자동 계산")
        for box in self.voltages:
            box.set_active(measured)
            box.ax.set_facecolor("white" if measured else "#eeeeee")
            box.text_disp.set_color("#222222" if measured else "#555555")
        try:
            if not measured:
                profile = design_profile(self.profile, self.read_design())
                for dbox, vbox in zip(self.distances, self.voltages):
                    value = expected_voltage_mv(profile, float(dbox.text), self.location())
                    vbox.text_disp.set_text(f"{value:.6g}")
            self.message.set_text("입력한 내용은 아직 적용되지 않았습니다. 아래 ‘적용’을 누르세요.")
            self.message.set_color("#925b00")
        except (ValueError, KeyError) as error:
            self.message.set_text("설계 입력 확인: " + str(error))
            self.message.set_color("firebrick")
        self.figure.canvas.draw_idle()

    def apply(self, event=None) -> bool:
        try:
            measured = self.mode == "실험값"
            voltages = [None if not box.text.strip() else float(box.text) for box in self.voltages] if measured else [None] * len(self.distances)
            distances = [float(dbox.text) if (not measured or v is not None) else None for dbox, v in zip(self.distances, voltages)]
            base = self.get_profile() if self.get_profile else self.profile
            profile = make_settings(base, self.read_design(), self.mode, self.location(), distances, voltages)
            self.on_apply(profile)
            self.profile = profile
            self.update_summary()
            self.message.set_text("적용 완료. 시뮬레이션 화면에서 ‘실행’을 누르세요.")
            self.message.set_color("darkgreen")
        except (ValueError, KeyError, OSError) as error:
            self.message.set_text("입력 확인: " + str(error))
            self.message.set_color("firebrick")
            self.figure.canvas.draw_idle()
            return False
        self.figure.canvas.draw_idle()
        return True

    def reset_to_defaults(self, event=None) -> bool:
        try:
            profile = default_settings()
            self.on_apply(profile)
            self.profile = profile
            self.restoring = True
            self.mode = "예상값"
            self.mode_radio.set_active(0)
            values = design_values(profile)
            for key, box in self.boxes.items():
                box.set_val(f"{values[key]:.6g}")
            rows = settings_rows(profile)
            for i, box in enumerate(self.distances):
                box.set_val(f"{rows[i][0] if i < len(rows) else 100 + i * 10:g}")
            self.measured_cache = [""] * len(self.voltages)
            self.restoring = False
            self.update_summary()
            self.refresh_results()
            self.message.set_text("예상값 모드와 설계 조건을 최초 기본값으로 초기화했습니다.")
            self.message.set_color("darkgreen")
        except (ValueError, KeyError, OSError) as error:
            self.restoring = False
            self.message.set_text(str(error))
            self.message.set_color("firebrick")
            return False
        self.figure.canvas.draw_idle()
        return True

    def show(self) -> None:
        plt.show(block=False)
