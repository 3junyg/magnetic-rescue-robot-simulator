from __future__ import annotations

import csv
import json
from copy import deepcopy
from pathlib import Path

import numpy as np

from simulation.coil_circuit_v2_1 import coaxial_mutual


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE = ROOT / "calibration" / "coil_profile_v2_1.json"


class MICalibration:
    def __init__(self, profile_path: str | Path | None = None, profile_overrides: dict | None = None) -> None:
        self.path = Path(profile_path) if profile_path else DEFAULT_PROFILE
        self.profile = json.loads(self.path.read_text(encoding="utf-8-sig"))
        if profile_overrides is not None:
            self.profile.update(deepcopy(profile_overrides))
        for key in ("frequency_hz", "tx_radius_m", "rx_radius_m", "tx_turns", "rx_turns", "tx_current_rms_a", "receiver_gain", "receiver_max_ac_rms_v", "world_unit_m", "detector_input_v_per_unit", "target_core_m"):
            value = float(self.profile[key])
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"Invalid positive MI parameter: {key}")
        for key in ("receiver_output_noise_rms_v", "tx_rx_offset_m", "wire_core_m", "human_polarizability_m3", "metal_polarizability_m3", "human_response_time_s", "metal_response_time_s"):
            value = float(self.profile[key])
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"Invalid nonnegative MI parameter: {key}")
        if not 0 <= self.profile["primary_cancellation_fraction"] <= 1:
            raise ValueError("primary_cancellation_fraction must be between 0 and 1")
        if int(self.profile["coil_segments"]) < 16:
            raise ValueError("coil_segments must be at least 16")
        if not np.isfinite(float(self.profile["demodulation_phase_deg"])):
            raise ValueError("demodulation_phase_deg must be finite")
        self.distance_m = np.empty(0)
        self.correction = np.empty(0)
        self.measured_transfer = np.empty(0)
        self.accepted_rows = 0
        self.excluded_rows = 0
        self.source = "provisional_theory"
        mode = self.profile.get("input_mode")
        if mode not in (None, "theory", "measured"):
            raise ValueError("input_mode must be theory or measured")
        if mode == "theory":
            pass
        elif self.profile.get("manual_transfer", {}).get("enabled", False):
            self._load_manual()
            if mode == "measured" and self.source != "measured_transfer":
                raise ValueError("실험값 모드에는 실제 측정 결과를 입력하세요.")
        elif mode == "measured":
            raise ValueError("실험값 모드에 결과가 없습니다.")
        else:
            self._load_measurements()
            if len(self.distance_m):
                self.source = "measured_transfer"
        self.profile["source"] = self.source

    def _load_manual(self) -> None:
        manual = self.profile["manual_transfer"]
        current = float(manual["reference_current_rms_a"])
        rows = np.asarray(manual["rows"], dtype=float)
        provenance = manual.get("provenance", "assumed")
        location = manual.get("voltage_location", "coil_input")
        if provenance not in ("assumed", "measured"):
            raise ValueError("입력 자료는 예상값 또는 실험값으로 구분해야 합니다.")
        if location not in ("coil_input", "amplifier_output"):
            raise ValueError("전압을 측정한 위치를 선택하세요.")
        gain = float(manual.get("measurement_gain", self.profile["receiver_gain"]))
        if not np.isfinite(gain) or gain <= 0:
            raise ValueError("측정 당시 수신 확대 배율은 0보다 커야 합니다.")
        if provenance == "measured":
            frequency = float(manual.get("frequency_hz", self.profile["frequency_hz"]))
            if not np.isfinite(frequency) or not np.isclose(frequency, self.profile["frequency_hz"], rtol=0.000001):
                raise ValueError("실험 주파수와 송신 주파수가 다릅니다. 같은 주파수의 실험값을 입력하세요.")
            if manual.get("relay_count", 0) != 0 or manual.get("rx_angle_deg", 0.0) != 0.0:
                raise ValueError("현재 실험값 입력은 중계 없음·정렬 0도 자료에만 적용됩니다.")
            if manual.get("medium", self.profile["measurement_medium"]) != self.profile["measurement_medium"]:
                raise ValueError("실험 매질과 현재 프로파일의 매질이 다릅니다.")
        if not np.isfinite(current) or current <= 0:
            raise ValueError("측정/가정 당시 전류는 0보다 커야 합니다.")
        if rows.ndim != 2 or rows.shape[1] != 2 or len(rows) < 2:
            raise ValueError("서로 다른 거리의 전압을 최소 두 쌍 입력하세요.")
        if not np.isfinite(rows).all() or np.any(rows <= 0):
            raise ValueError("거리와 전압은 0보다 큰 숫자여야 합니다. 측정하지 않은 행은 비워 두세요.")
        rows = rows[np.argsort(rows[:, 0])]
        if np.any(np.diff(rows[:, 0]) <= 0):
            raise ValueError("입력한 거리가 중복되었습니다.")
        self.distance_m = rows[:, 0].copy()
        self.measured_transfer = rows[:, 1] / current / (gain if location == "amplifier_output" else 1.0)
        self.correction = self.measured_transfer / np.array([self.direct_transfer(d) for d in self.distance_m])
        self.source = "measured_transfer" if provenance == "measured" else "manual_assumption"
        self.accepted_rows = len(rows) if provenance == "measured" else 0

    def direct_transfer(self, distance_m: float) -> float:
        p = self.profile
        return 2.0 * np.pi * p["frequency_hz"] * coaxial_mutual(p["tx_radius_m"], p["rx_radius_m"], float(distance_m), int(p["tx_turns"]), int(p["rx_turns"]))

    def _load_measurements(self) -> None:
        path = self.path.parent / self.profile["measurement_file"]
        if not path.is_file():
            raise FileNotFoundError(f"MI measurement file not found: {path}")
        groups: dict[float, list[float]] = {}
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            required = {"run_id", "repeat_id", "provenance", "medium", "frequency_hz", "distance_m", "rx_angle_deg", "relay_count", "tx_current_rms_a", "rx_voltage_rms_v", "voltage_location", "receiver_gain", "valid"}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError("MI CSV header is incomplete")
            for number, row in enumerate(reader, start=2):
                if not any((value or "").strip() for value in row.values()):
                    continue
                if row["valid"].strip().lower() not in ("1", "true", "yes"):
                    self.excluded_rows += 1
                    continue
                try:
                    frequency = float(row["frequency_hz"])
                    relay_count = int(row["relay_count"])
                    angle = float(row["rx_angle_deg"])
                    distance = float(row["distance_m"])
                    current = float(row["tx_current_rms_a"])
                    voltage = float(row["rx_voltage_rms_v"])
                    gain = float(row["receiver_gain"])
                    if not np.isfinite([frequency, angle, distance, current, voltage, gain]).all() or min(distance, current, gain) <= 0 or voltage < 0:
                        raise ValueError("invalid numeric values")
                    if row["provenance"] != "measured":
                        raise ValueError("valid experimental rows must declare provenance=measured")
                    if row["voltage_location"] not in ("coil_input", "amplifier_output"):
                        raise ValueError("voltage_location must be coil_input or amplifier_output")
                    if relay_count != 0 or abs(angle) > 1e-6 or row["medium"] != self.profile["measurement_medium"] or not np.isclose(frequency, self.profile["frequency_hz"], rtol=0.01):
                        self.excluded_rows += 1
                        continue
                    voltage /= gain if row["voltage_location"] == "amplifier_output" else 1.0
                    groups.setdefault(distance, []).append(voltage / current)
                    self.accepted_rows += 1
                except (ValueError, TypeError) as error:
                    raise ValueError(f"MI measurement CSV line {number}: {error}") from error
        if groups:
            if len(groups) < 2:
                raise ValueError("Experimental transfer calibration needs at least two distinct direct-link distances")
            self.distance_m = np.array(sorted(groups))
            self.measured_transfer = np.array([np.mean(groups[distance]) for distance in self.distance_m])
            self.correction = self.measured_transfer / np.array([self.direct_transfer(distance) for distance in self.distance_m])
            if np.any(self.correction <= 0):
                raise ValueError("Zero/below-noise measurements must be marked invalid")

    def factors(self, distances_m: np.ndarray) -> np.ndarray:
        distances = np.asarray(distances_m, dtype=float)
        if not len(self.distance_m):
            return np.ones_like(distances)
        inside = (distances >= self.distance_m[0]) & (distances <= self.distance_m[-1])
        result = np.ones_like(distances)
        result[inside] = np.interp(distances[inside], self.distance_m, self.correction)
        return result

    def status(self) -> dict:
        active_range = self.distance_m[[0, -1]].tolist() if len(self.distance_m) else None
        return {"version": "2.1", "source": self.source, "frequency_hz": self.profile["frequency_hz"], "receiver_gain": self.profile["receiver_gain"], "tx_current_rms_a": self.profile["tx_current_rms_a"], "accepted_rows": self.accepted_rows, "excluded_rows": self.excluded_rows, "measured_range_m": active_range if self.source == "measured_transfer" else None, "manual_range_m": active_range if self.source == "manual_assumption" else None, "outside_range": "theory_fallback_no_extrapolation", "target_response": "synthetic_unvalidated", "cnn_status": "V2_WEIGHTS_NOT_VALIDATED_ON_V2_1", "scan_status": "synthetic_virtual_grid", "profile_path": str(self.path), "measurement_path": str(self.path.parent / self.profile["measurement_file"])}
