from dataclasses import dataclass

import numpy as np


@dataclass
class LocalizationMatch:
    marker_index: int
    person_index: int
    distance: float
    score: float


@dataclass
class LocalizationEvaluation:
    matches: list[LocalizationMatch]
    false_positives: int
    missed_people: int
    mean_distance: float | None
    localization_score: float
    detection_rate: float


def evaluate_localization(markers, people, maximum_distance: float = 8.0) -> LocalizationEvaluation:
    pairs = []
    for marker_index, marker in enumerate(markers):
        for person_index, person in enumerate(people):
            distance = float(np.linalg.norm(marker.position - person.position))
            if distance <= maximum_distance:
                pairs.append((distance, marker_index, person_index))
    pairs.sort()
    used_markers = set()
    used_people = set()
    matches = []
    for distance, marker_index, person_index in pairs:
        if marker_index in used_markers or person_index in used_people:
            continue
        score = 100.0 * max(0.0, 1.0 - distance / maximum_distance)
        matches.append(LocalizationMatch(marker_index, person_index, distance, score))
        used_markers.add(marker_index)
        used_people.add(person_index)
    false_positives = len(markers) - len(matches)
    missed_people = len(people) - len(matches)
    denominator = max(1, len(markers), len(people))
    localization_score = sum(match.score for match in matches) / denominator
    detection_rate = len(matches) / max(1, len(people))
    mean_distance = float(np.mean([match.distance for match in matches])) if matches else None
    return LocalizationEvaluation(matches, false_positives, missed_people, mean_distance, localization_score, detection_rate)
