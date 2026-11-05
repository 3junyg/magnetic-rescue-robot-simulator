from __future__ import annotations

from collections.abc import Callable

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons

from simulation.relay_settings_v2_1 import relay_values, relay_summary, with_relay
from ui.mi_parameters_v2_1 import MITextBox


class RelayParameterWindow:
    def __init__(self, get_profile: Callable[[], dict], on_apply: Callable[[dict], None]) -> None:
        self.get_profile = get_profile
        self.on_apply = on_apply
        settings = relay_values(get_profile())
        self.figure = plt.figure(figsize=(9, 7))
        self.figure.canvas.manager.set_window_title("V2.1 코일 중계 설정")
        self.figure.text(0.07, 0.93, "코일 중계 설정", fontsize=18, weight="bold")
        self.figure.text(0.07, 0.86, "송신 → 같은 간격의 공진 중계 코일 → 수신\n예상 회로식으로 항상 계산합니다. 실제 배치 지도는 만들지 않습니다.", fontsize=11)
        self.enabled = RadioButtons(self.figure.add_axes([0.07, 0.70, 0.35, 0.10]), ["사용 안 함", "사용"], active=int(settings["enabled"]))
        self.axis = RadioButtons(self.figure.add_axes([0.58, 0.67, 0.32, 0.15]), ["X", "Y", "Z"], active=("X", "Y", "Z").index(settings["axis"]))
        self.figure.text(0.58, 0.83, "전달 방향과 정렬할 코일 축", fontsize=10)
        self.figure.text(0.07, 0.61, "중계 장치 개수 (1~8개)", fontsize=11)
        self.count = MITextBox(self.figure.add_axes([0.66, 0.59, 0.24, 0.045]), "", initial=str(settings["count"]))
        self.figure.text(0.07, 0.53, "이웃한 코일 장치 중심 간격 (cm)", fontsize=11)
        self.spacing = MITextBox(self.figure.add_axes([0.66, 0.51, 0.24, 0.045]), "", initial=f'{settings["spacing_cm"]:g}')
        self.result = self.figure.text(0.07, 0.24, "", fontsize=11, linespacing=1.7)
        self.figure.text(0.07, 0.17, "3축 장치의 선택한 축 코일 쌍을 동축으로 연결한 등가 모델입니다.\n코일 수·거리에 따라 감소할 수도 있습니다. L/R/C는 실측 전 가정값입니다.", fontsize=9)
        self.message = self.figure.text(0.07, 0.10, "", fontsize=10)
        self.apply_button = Button(self.figure.add_axes([0.62, 0.03, 0.28, 0.05]), "적용")
        self.apply_button.on_clicked(self.apply)
        for widget in (self.count, self.spacing):
            widget.on_text_change(self.refresh)
        self.enabled.on_clicked(self.refresh)
        self.axis.on_clicked(self.refresh)
        self.refresh()

    def settings(self) -> dict:
        return {"enabled": self.enabled.value_selected == "사용", "count": float(self.count.text),
                "spacing_cm": float(self.spacing.text), "axis": self.axis.value_selected}

    def refresh(self, event=None) -> None:
        try:
            profile = with_relay(self.get_profile(), self.settings())
            summary = relay_summary(profile)
            self.result.set_text(
                f"송신–수신 전체 거리: {summary['distance_m'] * 100:g} cm\n"
                f"중계 없음: {summary['direct_rms_v'] * 1000:.6g} mV RMS\n"
                f"현재 설정: {summary['input_rms_v'] * 1000:.6g} mV RMS\n"
                f"수신 전압 배율: {summary['gain']:.4g}배 · 위상 변화: {summary['phase_deg']:.1f}°")
            self.message.set_text("미리보기입니다. 적용하면 반응 세기와 센서 범위만 바뀝니다.")
            self.message.set_color("#925b00")
        except (ValueError, KeyError, TypeError) as error:
            self.result.set_text("")
            self.message.set_text(str(error))
            self.message.set_color("firebrick")
        self.figure.canvas.draw_idle()

    def apply(self, event=None) -> bool:
        try:
            self.on_apply(with_relay(self.get_profile(), self.settings()))
            self.message.set_text("적용 완료. 시뮬레이션에서 실행을 누르세요.")
            self.message.set_color("darkgreen")
        except (ValueError, KeyError, TypeError) as error:
            self.message.set_text(str(error))
            self.message.set_color("firebrick")
            return False
        self.figure.canvas.draw_idle()
        return True

    def show(self) -> None:
        plt.show(block=False)
