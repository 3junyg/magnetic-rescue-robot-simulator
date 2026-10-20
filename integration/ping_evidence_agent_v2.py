import numpy as np


class PingEvidenceAgent:
    @staticmethod
    def eligible(probability: float, location_probability: float, settings) -> bool:
        return (
            probability >= settings.candidate_probability_threshold
            and location_probability >= settings.candidate_location_threshold
        )

    @staticmethod
    def support(probability: float, location_probability: float, dynamic_score: float) -> float:
        presence = np.clip((probability - 0.45) / 0.55, 0.0, 1.0)
        location = 0.45 + 0.55 * np.clip(location_probability, 0.0, 1.0)
        dynamics = 0.20 + 0.80 * np.clip((dynamic_score - 0.25) / 0.75, 0.0, 1.0)
        return float(presence * location * dynamics)

    @staticmethod
    def statistics(track) -> tuple[float, float, float, float, float]:
        samples = list(track.samples)
        positions = np.asarray([sample.position for sample in samples])
        observers = np.asarray([sample.observer_position for sample in samples])
        mean_probability = float(np.mean([sample.probability for sample in samples]))
        mean_location_probability = float(np.mean([sample.location_probability for sample in samples]))
        mean_dynamic_score = float(np.mean([sample.dynamic_score for sample in samples]))
        center = np.mean(positions, axis=0)
        stability = float(np.sqrt(np.mean(np.sum((positions - center) ** 2, axis=1))))
        observer_span = float(np.max(np.linalg.norm(observers - observers[0], axis=1)))
        return mean_probability, mean_location_probability, mean_dynamic_score, stability, observer_span


class PingConfirmationAgent:
    def __init__(self, evidence: PingEvidenceAgent) -> None:
        self.evidence = evidence

    def mode(self, track, settings, near_static_source: bool = False) -> str | None:
        mean_probability, mean_location_probability, mean_dynamic_score, stability, observer_span = self.evidence.statistics(track)
        recent = list(track.samples)[-settings.fast_hits:]
        strong_recent = len(recent) >= settings.fast_hits and min(sample.probability for sample in recent) >= settings.fast_probability
        dynamic_recent = len(recent) >= settings.fast_hits and min(sample.dynamic_score for sample in recent) >= settings.fast_dynamic_score
        normal_recent = list(track.samples)[-5:]
        dynamic_persistence = sum(sample.dynamic_score >= 0.50 for sample in normal_recent) >= settings.normal_dynamic_hits
        fast = (
            strong_recent
            and dynamic_recent
            and track.consecutive_hits >= settings.fast_hits
            and track.evidence >= settings.fast_evidence
            and mean_location_probability >= settings.fast_location_probability
            and mean_dynamic_score >= settings.fast_dynamic_score
            and stability <= settings.fast_stability
            and observer_span >= settings.fast_observer_span
        )
        normal = (
            track.hits >= settings.normal_hits
            and track.consecutive_hits >= settings.normal_consecutive_hits
            and dynamic_persistence
            and track.evidence >= settings.normal_evidence
            and mean_probability >= settings.normal_probability
            and mean_location_probability >= settings.candidate_location_threshold
            and mean_dynamic_score >= settings.normal_dynamic_score
            and stability <= settings.normal_stability
            and observer_span >= settings.normal_observer_span
        )
        close_recent = list(track.samples)[-2:]
        proximity = (
            len(close_recent) == 2
            and track.consecutive_hits >= 2
            and min(sample.probability for sample in close_recent) >= settings.proximity_probability
            and min(sample.location_probability for sample in close_recent) >= settings.proximity_location_probability
            and min(sample.dynamic_score for sample in close_recent) >= settings.proximity_dynamic_score
            and max(float(np.linalg.norm(track.position - sample.observer_position)) for sample in close_recent) <= settings.proximity_distance
            and stability <= settings.proximity_stability
            and observer_span >= settings.proximity_observer_span
        )
        if near_static_source:
            if (
                normal
                and track.hits >= 7
                and mean_location_probability >= 0.55
                and mean_dynamic_score >= 0.90
                and stability <= 2.5
                and observer_span >= 1.4
            ):
                return "STATIC_BACKGROUND_VERIFIED"
            return None
        if proximity:
            return "PROXIMITY"
        if fast:
            return "FAST"
        if normal:
            return "ACCUMULATED"
        return None
