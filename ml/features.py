import numpy as np

from config import SENSOR_RANGE


RADIAL_BINS = 4
ANGULAR_BINS = 8
CHANNELS = 8
FEATURE_SIZE = RADIAL_BINS * ANGULAR_BINS * CHANNELS


def extract_scan_features(scan, robot_position: np.ndarray, heading: float) -> np.ndarray:
    return extract_vector_features(scan.positions, scan.vectors, robot_position, heading)


def extract_vector_features(positions: np.ndarray, vectors: np.ndarray, robot_position: np.ndarray, heading: float, reference: np.ndarray | None = None) -> np.ndarray:
    if reference is None:
        reference = np.array([24.0, -7.0, 39.0])
    magnitudes = np.linalg.norm(vectors, axis=1)
    relative = positions - robot_position
    cosine = np.cos(heading)
    sine = np.sin(heading)
    local_x = cosine * relative[:, 0] + sine * relative[:, 1]
    local_y = -sine * relative[:, 0] + cosine * relative[:, 1]
    distances = np.sqrt(local_x ** 2 + local_y ** 2)
    angles = (np.arctan2(local_y, local_x) + 2.0 * np.pi) % (2.0 * np.pi)
    radial_indices = np.minimum(RADIAL_BINS - 1, (distances / SENSOR_RANGE * RADIAL_BINS).astype(int))
    angular_indices = np.minimum(ANGULAR_BINS - 1, (angles / (2.0 * np.pi) * ANGULAR_BINS).astype(int))
    forward = cosine * vectors[:, 0] + sine * vectors[:, 1]
    lateral = -sine * vectors[:, 0] + cosine * vectors[:, 1]
    values = np.column_stack([forward, lateral, vectors[:, 2], magnitudes])
    features = np.zeros((RADIAL_BINS, ANGULAR_BINS, CHANNELS), dtype=np.float32)
    counts = np.zeros((RADIAL_BINS, ANGULAR_BINS), dtype=np.int32)
    anomaly = np.linalg.norm(vectors - reference, axis=1)
    magnitude_square = magnitudes ** 2
    for radial, angular, value, anomaly_value, square_value in zip(radial_indices, angular_indices, values, anomaly, magnitude_square):
        features[radial, angular, :4] += value
        features[radial, angular, 4] += square_value
        features[radial, angular, 5] = max(features[radial, angular, 5], anomaly_value)
        features[radial, angular, 6] += anomaly_value
        features[radial, angular, 7] = max(features[radial, angular, 7], value[3])
        counts[radial, angular] += 1
    occupied = counts > 0
    divisors = counts[occupied][:, None]
    features[occupied, :4] /= divisors
    mean_square = features[occupied, 4] / counts[occupied]
    mean_magnitude = features[occupied, 3]
    features[occupied, 4] = np.sqrt(np.maximum(0.0, mean_square - mean_magnitude ** 2))
    features[occupied, 6] /= counts[occupied]
    return features


def local_motion(previous_position: np.ndarray, current_position: np.ndarray, previous_heading: float, current_heading: float) -> np.ndarray:
    displacement = current_position - previous_position
    cosine = np.cos(previous_heading)
    sine = np.sin(previous_heading)
    forward = cosine * displacement[0] + sine * displacement[1]
    lateral = -sine * displacement[0] + cosine * displacement[1]
    heading_change = (current_heading - previous_heading + np.pi) % (2.0 * np.pi) - np.pi
    return np.array([forward, lateral, heading_change], dtype=np.float32)
