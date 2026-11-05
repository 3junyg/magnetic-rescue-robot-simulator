from __future__ import annotations

from functools import lru_cache
import math

import numpy as np


MU0 = 4.0 * math.pi * 1e-7


@lru_cache(maxsize=2048)
def coaxial_mutual(radius_a: float, radius_b: float, distance: float, turns_a: int, turns_b: int) -> float:
    if min(radius_a, radius_b, distance, turns_a, turns_b) <= 0:
        raise ValueError("Coil dimensions, turns and center distance must be positive")
    count = 512
    angle = (np.arange(count) + 0.5) * (2.0 * np.pi / count)
    denominator = np.sqrt(distance**2 + radius_a**2 + radius_b**2 - 2.0 * radius_a * radius_b * np.cos(angle))
    integral = float(np.mean(np.cos(angle) / denominator) * 2.0 * np.pi)
    return MU0 * turns_a * turns_b * radius_a * radius_b * integral / 2.0


def coil_field(points_m: np.ndarray, center_m: np.ndarray, normal: np.ndarray, radius_m: float, turns: int, current_a: float, segments: int = 48, wire_core_m: float = 0.001) -> np.ndarray:
    points = np.asarray(points_m, dtype=float).reshape(-1, 3)
    normal = np.asarray(normal, dtype=float)
    normal = normal / np.linalg.norm(normal)
    reference = np.array([0.0, 0.0, 1.0]) if abs(normal[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    first = np.cross(reference, normal)
    first /= np.linalg.norm(first)
    second = np.cross(normal, first)
    angle = (np.arange(segments) + 0.5) * 2.0 * np.pi / segments
    ring = radius_m * (np.cos(angle)[:, None] * first + np.sin(angle)[:, None] * second)
    dl = radius_m * 2.0 * np.pi / segments * (-np.sin(angle)[:, None] * first + np.cos(angle)[:, None] * second)
    relative = points[:, None, :] - (np.asarray(center_m) + ring)[None, :, :]
    denominator = np.maximum(np.sum(relative * relative, axis=2), wire_core_m**2) ** 1.5
    return MU0 * turns * current_a / (4.0 * np.pi) * np.sum(np.cross(dl[None, :, :], relative) / denominator[:, :, None], axis=1)


def relay_prediction(profile: dict, distance_m: float, relay_count: int = 0, axis: str = "Y") -> dict:
    if isinstance(relay_count, bool) or int(relay_count) != relay_count or not 0 <= relay_count <= 8:
        raise ValueError("relay_count must be an integer from 0 to 8")
    relay_count = int(relay_count)
    if not np.isfinite(distance_m) or distance_m <= 0:
        raise ValueError("distance_m must be positive")
    relay = profile["relay_axes"][axis]
    frequency = float(profile["frequency_hz"])
    omega = 2.0 * np.pi * frequency
    current = float(profile["tx_current_rms_a"])
    positions = np.arange(1, relay_count + 1) * distance_m / (relay_count + 1)
    offsets = np.array([-0.5, 0.5]) * relay["separation_m"]
    if relay_count and distance_m / (relay_count + 1) <= 2.0 * max(item["radius_m"] for item in profile["relay_axes"].values()) + 0.02:
        raise ValueError("Relay frames need more separation; increase total distance")

    def coupling_to_single(center: float, single: float, radius: float, turns: int) -> float:
        return sum(coaxial_mutual(relay["radius_m"], radius, abs(center + offset - single), relay["turns_per_coil"], turns) for offset in offsets)

    impedance = relay["resistance_ohm"] + 1j * (omega * relay["inductance_h"] - 1.0 / (omega * relay["capacitance_f"]))
    matrix = np.eye(relay_count, dtype=complex) * impedance
    for i in range(relay_count):
        for j in range(i + 1, relay_count):
            mutual = sum(coaxial_mutual(relay["radius_m"], relay["radius_m"], abs(positions[i] + a - positions[j] - b), relay["turns_per_coil"], relay["turns_per_coil"]) for a in offsets for b in offsets)
            matrix[i, j] = matrix[j, i] = 1j * omega * mutual
    induced = np.array([-1j * omega * coupling_to_single(center, 0.0, profile["tx_radius_m"], profile["tx_turns"]) * current for center in positions])
    relay_currents = np.linalg.solve(matrix, induced) if relay_count else np.empty(0, dtype=complex)
    flux = coaxial_mutual(profile["tx_radius_m"], profile["rx_radius_m"], distance_m, profile["tx_turns"], profile["rx_turns"]) * current
    for center, relay_current in zip(positions, relay_currents):
        flux += coupling_to_single(center, distance_m, profile["rx_radius_m"], profile["rx_turns"]) * relay_current
    voltage_complex = -1j * omega * flux
    direct_complex = -1j * omega * coaxial_mutual(profile["tx_radius_m"], profile["rx_radius_m"], distance_m, profile["tx_turns"], profile["rx_turns"]) * current
    ratio = voltage_complex / direct_complex if relay_count else complex(1.0)
    voltage = abs(voltage_complex)
    return {"distance_m": distance_m, "relay_count": relay_count, "axis": axis, "input_rms_v": float(voltage), "output_rms_v": float(voltage * profile["receiver_gain"]), "direct_rms_v": float(abs(direct_complex)), "gain": float(abs(ratio)), "phase_deg": float(np.angle(ratio, deg=True)), "ratio_real": float(ratio.real), "ratio_imag": float(ratio.imag), "transfer_input_v_per_a": float(voltage / current), "relay_current_rms_a": np.abs(relay_currents).tolist(), "source": "provisional_circuit_theory"}
