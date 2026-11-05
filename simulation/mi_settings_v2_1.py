from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path

from simulation.coil_circuit_v2_1 import coaxial_mutual
from simulation.mi_calibration_v2_1 import MICalibration


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DISTANCES_CM = [20.0, 30.0, 40.0, 50.0, 60.0, 80.0]
DESIGN_PARAMETERS = (
    ("frequency_hz", "송신 주파수 (Hz)", 1.0),
    ("tx_current_rms_a", "송신 전류 (mA RMS)", 1000.0),
    ("tx_radius_m", "송신 코일 반지름 (cm)", 100.0),
    ("tx_turns", "송신 코일 감은 횟수 (회)", 1.0),
    ("rx_radius_m", "수신 코일 반지름 (cm)", 100.0),
    ("rx_turns", "수신 코일 감은 횟수 (회)", 1.0),
)


def design_values(profile: dict) -> dict:
    return {key: float(profile[key]) * scale for key, _, scale in DESIGN_PARAMETERS}


def default_settings() -> dict:
    path = ROOT / "calibration" / "default_profile_v2_1.json"
    return MICalibration(profile_overrides=json.loads(path.read_text(encoding="utf-8"))).profile


def design_profile(base: dict, values: dict) -> dict:
    profile = deepcopy(base)
    for key, label, scale in DESIGN_PARAMETERS:
        try:
            value = float(values[key])
        except (ValueError, TypeError) as error:
            raise ValueError(f"{label}: 숫자를 입력하세요.") from error
        if not math.isfinite(value) or value < 0 or (value == 0 and key != "receiver_output_noise_rms_v"):
            raise ValueError(f"{label}: {'0 이상의' if key == 'receiver_output_noise_rms_v' else '0보다 큰'} 숫자를 입력하세요.")
        if key.endswith("_turns") and not value.is_integer():
            raise ValueError(f"{label}: 정수로 입력하세요.")
        profile[key] = int(value) if key.endswith("_turns") else value / scale
    return profile


def expected_voltage_mv(profile: dict, distance_cm: float, location: str = "coil_input") -> float:
    distance = float(distance_cm) / 100.0
    if not math.isfinite(distance) or distance <= 0:
        raise ValueError("거리는 0보다 큰 숫자여야 합니다.")
    if location not in ("coil_input", "amplifier_output"):
        raise ValueError("전압 위치를 선택하세요.")
    mutual = coaxial_mutual(profile["tx_radius_m"], profile["rx_radius_m"], distance,
                            profile["tx_turns"], profile["rx_turns"])
    voltage = 2.0 * math.pi * profile["frequency_hz"] * mutual * profile["tx_current_rms_a"]
    return voltage * (profile["receiver_gain"] if location == "amplifier_output" else 1.0) * 1000.0


def make_settings(base: dict, values: dict, mode: str, location: str,
                  distances_cm: list[float | None], voltages_mv: list[float | None], signal_mode: str | None = None) -> dict:
    if mode not in ("예상값", "실험값"):
        raise ValueError("예상값 또는 실험값을 선택하세요.")
    if location not in ("coil_input", "amplifier_output"):
        raise ValueError("전압 위치를 선택하세요.")
    if not 1 <= len(distances_cm) <= 64 or len(distances_cm) != len(voltages_mv):
        raise ValueError("거리와 결과의 행 수를 확인하세요.")
    profile = design_profile(base, values)
    profile["signal_mode"] = "scaled_v2" if signal_mode is None else signal_mode
    if profile["signal_mode"] not in ("v2", "scaled_v2", "calibrated_mi"):
        raise ValueError("신호 설정을 확인하세요.")
    rows = []
    for distance, voltage in zip(distances_cm, voltages_mv):
        if mode == "실험값" and voltage is None:
            continue
        if distance is None:
            raise ValueError("결과값에 해당하는 거리를 입력하세요.")
        computed = expected_voltage_mv(profile, distance, location)
        result = computed if mode == "예상값" else float(voltage)
        if not math.isfinite(result) or result <= 0:
            raise ValueError("수신 전압은 0보다 큰 숫자여야 합니다.")
        rows.append([float(distance) / 100.0, result / 1000.0])
    if len(rows) < (2 if mode == "실험값" else 1):
        raise ValueError("실험값은 서로 다른 거리 2개 이상, 예상값은 거리 1개 이상 입력하세요.")
    if len({d for d, _ in rows}) != len(rows):
        raise ValueError("거리가 중복되었습니다.")
    profile["input_mode"] = "measured" if mode == "실험값" else "theory"
    profile["voltage_location"] = location
    profile["design_distances_cm"] = [d * 100 for d, _ in rows]
    profile["manual_transfer"] = {
        "enabled": mode == "실험값",
        "provenance": "measured" if mode == "실험값" else "assumed",
        "reference_current_rms_a": profile["tx_current_rms_a"],
        "frequency_hz": profile["frequency_hz"],
        "measurement_gain": profile["receiver_gain"],
        "voltage_location": location,
        "relay_count": 0,
        "rx_angle_deg": 0.0,
        "medium": profile["measurement_medium"],
        "rows": rows,
    }
    return MICalibration(profile_overrides=profile).profile


def settings_rows(profile: dict) -> list[tuple[float, float | None]]:
    measured = profile.get("input_mode") == "measured" or (
        profile.get("input_mode") is None and profile.get("manual_transfer", {}).get("enabled")
        and profile.get("manual_transfer", {}).get("provenance") == "measured")
    if measured:
        rows = {float(d) * 100: float(v) * 1000 for d, v in profile["manual_transfer"]["rows"]}
        for d in DEFAULT_DISTANCES_CM:
            if len(rows) >= 6:
                break
            rows.setdefault(d, None)
        return sorted(rows.items())
    location = profile.get("voltage_location", "coil_input")
    return [(d, expected_voltage_mv(profile, d, location)) for d in profile.get("design_distances_cm", DEFAULT_DISTANCES_CM)]
