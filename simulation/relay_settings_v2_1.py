from __future__ import annotations

from copy import deepcopy
import math

from simulation.coil_circuit_v2_1 import relay_prediction


DEFAULT_RELAY = {"enabled": False, "count": 1, "spacing_cm": 30.0, "axis": "Y"}


def relay_values(profile: dict) -> dict:
    result = {**DEFAULT_RELAY, **profile.get("relay_settings", {})}
    if not isinstance(result["enabled"], bool):
        raise ValueError("코일 중계 사용 여부를 확인하세요.")
    count = float(result["count"])
    spacing = float(result["spacing_cm"])
    if not math.isfinite(count) or not count.is_integer() or not 1 <= count <= 8:
        raise ValueError("중계 장치 개수는 1~8개의 정수로 입력하세요.")
    if not math.isfinite(spacing) or spacing <= 0:
        raise ValueError("코일 사이 거리는 0보다 큰 숫자로 입력하세요.")
    if result["axis"] not in ("X", "Y", "Z"):
        raise ValueError("중계 코일 축은 X, Y, Z 중 하나를 선택하세요.")
    result.update(count=int(count), spacing_cm=spacing)
    return result


def relay_summary(profile: dict) -> dict:
    settings = relay_values(profile)
    distance = (settings["count"] + 1) * settings["spacing_cm"] / 100
    count = settings["count"] if settings["enabled"] else 0
    prediction = relay_prediction(profile, distance, count, settings["axis"])
    prediction.update(settings=settings, model="coaxial_equal_spacing_equivalent_link",
                      electronics_gain=float(profile["receiver_gain"]))
    return prediction


def with_relay(profile: dict, settings: dict) -> dict:
    result = deepcopy(profile)
    result["relay_settings"] = dict(settings)
    result["relay_settings"] = relay_values(result)
    relay_summary(result)
    return result


def relay_factor(profile: dict) -> complex:
    summary = relay_summary(profile)
    return complex(summary["ratio_real"], summary["ratio_imag"])
