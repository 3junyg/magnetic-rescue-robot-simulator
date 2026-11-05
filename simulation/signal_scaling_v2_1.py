from __future__ import annotations

import math

import numpy as np

from config_v2_1 import BOARD_HEIGHT, BOARD_WIDTH, PERCEPTION_CELL_SIZE, SENSOR_RANGE
from simulation.coil_circuit_v2_1 import coaxial_mutual
from simulation.mi_calibration_v2_1 import MICalibration
from simulation.relay_settings_v2_1 import relay_summary


REFERENCE_DISTANCE_M = 0.30
REFERENCE_VOLTAGE_V = 2 * math.pi * 2000 * coaxial_mutual(0.05, 0.05, REFERENCE_DISTANCE_M, 100, 100) * 0.05
RANGE_EXPONENT = 3.0


def signal_scaling(calibration: MICalibration) -> dict:
    p = calibration.profile
    theory_v = calibration.direct_transfer(REFERENCE_DISTANCE_M) * p["tx_current_rms_a"]
    correction = float(np.median(calibration.correction)) if len(calibration.correction) else 1.0
    relay = relay_summary(p)
    factor = float(theory_v / REFERENCE_VOLTAGE_V * correction * relay["gain"])
    if not math.isfinite(factor) or factor <= 0:
        raise ValueError("신호 배율은 유한한 양수여야 합니다.")
    raw_range = SENSOR_RANGE * factor ** (1 / RANGE_EXPONENT)
    radius = float(np.clip(raw_range, PERCEPTION_CELL_SIZE, math.hypot(BOARD_WIDTH, BOARD_HEIGHT)))
    return {"strength_factor": factor, "sensor_range": radius, "base_sensor_range": SENSOR_RANGE,
            "raw_sensor_range": raw_range, "range_clipped": radius != raw_range,
            "range_exponent": RANGE_EXPONENT, "reference_distance_m": REFERENCE_DISTANCE_M,
            "transfer_correction": correction, "relay_gain": relay["gain"],
            "model": "v2_amplitude_and_range_only", "range_relation": "synthetic_not_measured"}
