from collections import deque
from dataclasses import dataclass, field

import numpy as np

from integration.ping_evidence_agent_v2 import PingConfirmationAgent, PingEvidenceAgent
from integration.detection_candidates_v2 import DetectionProposal


@dataclass
class PingDecision:
    position: np.ndarray
    confidence: float
    mode: str


@dataclass
class DetectionSample:
    position: np.ndarray
    observer_position: np.ndarray
    probability: float
    location_probability: float
    dynamic_score: float
    timestamp: float


@dataclass
class CandidateTrack:
    position: np.ndarray
    velocity: np.ndarray
    evidence: float
    hits: int
    consecutive_hits: int
    last_time: float
    samples: deque[DetectionSample] = field(default_factory=lambda: deque(maxlen=8))
    emitted: bool = False


class PingDecisionAgent:
    def __init__(self) -> None:
        self.candidate_probability_threshold = 0.47
        self.candidate_location_threshold = 0.39
        self.association_radius = 7.0
        self.single_person_suppression_radius = 16.0
        self.multi_person_suppression_radius = 12.0
        self.track_decay_seconds = 5.0
        self.track_expiry_seconds = 8.0
        self.minimum_interval = 0.14
        self.fast_probability = 0.97
        self.fast_location_probability = 0.48
        self.fast_dynamic_score = 0.75
        self.fast_hits = 2
        self.fast_evidence = 0.95
        self.fast_stability = 2.0
        self.fast_observer_span = 0.50
        self.proximity_probability = 0.70
        self.proximity_location_probability = 0.39
        self.proximity_dynamic_score = 2.50
        self.proximity_distance = 8.0
        self.proximity_stability = 2.0
        self.proximity_observer_span = 1.0
        self.normal_probability = 0.69
        self.normal_dynamic_score = 0.56
        self.normal_hits = 3
        self.normal_consecutive_hits = 2
        self.normal_evidence = 1.60
        self.normal_dynamic_hits = 3
        self.normal_stability = 3.5
        self.normal_observer_span = 0.50
        self.evidence_agent = PingEvidenceAgent()
        self.confirmation_agent = PingConfirmationAgent(self.evidence_agent)
        self.tracks: list[CandidateTrack] = []
        self.emitted_positions: list[np.ndarray] = []
        self.ignored_positions: list[np.ndarray] = []
        self.estimated_count_history: deque[int] = deque(maxlen=5)
        self.last_update_time: float | None = None

    def reset(self) -> None:
        self.tracks.clear()
        self.emitted_positions.clear()
        self.ignored_positions.clear()
        self.estimated_count_history.clear()
        self.last_update_time = None

    @property
    def clusters(self) -> list[CandidateTrack]:
        return self.tracks

    def ignore_position(self, position: np.ndarray, radius: float = 8.0) -> None:
        point = np.asarray(position, dtype=float).copy()
        if not any(float(np.linalg.norm(point - saved)) <= radius for saved in self.ignored_positions):
            self.ignored_positions.append(point)

    def _age_tracks(self, timestamp: float) -> None:
        if self.last_update_time is None:
            self.last_update_time = timestamp
            return
        elapsed = max(0.0, timestamp - self.last_update_time)
        decay = float(np.exp(-elapsed / self.track_decay_seconds))
        for track in self.tracks:
            track.evidence *= decay
        self.tracks = [
            track
            for track in self.tracks
            if not track.emitted and timestamp - track.last_time <= self.track_expiry_seconds
        ]
        self.last_update_time = timestamp

    def _reset_continuity(self) -> None:
        for track in self.tracks:
            track.consecutive_hits = 0

    def _is_suppressed(self, position: np.ndarray, estimated_count: int) -> bool:
        multiple_people = estimated_count >= 2 and sum(count >= 2 for count in self.estimated_count_history) >= 3
        radius = self.multi_person_suppression_radius if multiple_people else self.single_person_suppression_radius
        return any(np.linalg.norm(position - saved) <= radius for saved in self.emitted_positions) or any(
            np.linalg.norm(position - saved) <= 8.0 for saved in self.ignored_positions
        )

    def _find_track(self, position: np.ndarray, timestamp: float, excluded: set[int] | None = None) -> CandidateTrack | None:
        available = [track for track in self.tracks if excluded is None or id(track) not in excluded]
        if not available:
            return None
        predictions = []
        for track in available:
            elapsed = max(0.0, timestamp - track.last_time)
            predictions.append(track.position + track.velocity * elapsed)
        distances = [float(np.linalg.norm(position - prediction)) for prediction in predictions]
        index = int(np.argmin(distances))
        return available[index] if distances[index] <= self.association_radius else None

    def process(
        self,
        timestamp: float,
        observer_position: np.ndarray,
        candidate_position: np.ndarray | None,
        probability: float,
        location_probability: float,
        estimated_count: int = 1,
        dynamic_score: float = 0.0,
        static_source_positions: tuple[np.ndarray, ...] = (),
    ) -> PingDecision | None:
        proposals = [] if candidate_position is None else [
            DetectionProposal(
                np.asarray(candidate_position, dtype=float),
                location_probability,
                dynamic_score,
                -1,
            )
        ]
        return self.process_candidates(
            timestamp,
            observer_position,
            proposals,
            probability,
            estimated_count,
            static_source_positions,
        )

    def process_candidates(
        self,
        timestamp: float,
        observer_position: np.ndarray,
        proposals: list[DetectionProposal],
        probability: float,
        estimated_count: int = 1,
        static_source_positions: tuple[np.ndarray, ...] = (),
    ) -> PingDecision | None:
        self._age_tracks(timestamp)
        self.estimated_count_history.append(estimated_count)
        observer = np.asarray(observer_position, dtype=float)
        matched_tracks: set[int] = set()
        confirmed: list[tuple[float, CandidateTrack, str]] = []
        accepted = False
        for proposal in proposals:
            location_probability = proposal.location_probability
            dynamic_score = proposal.dynamic_score
            if not self.evidence_agent.eligible(probability, location_probability, self):
                continue
            position = np.asarray(proposal.position, dtype=float)
            if self._is_suppressed(position, estimated_count):
                continue
            accepted = True
            track = self._find_track(position, timestamp, matched_tracks)
            if track is None:
                sample = DetectionSample(position.copy(), observer.copy(), probability, location_probability, dynamic_score, timestamp)
                track = CandidateTrack(
                    position.copy(),
                    np.zeros(2),
                    self.evidence_agent.support(probability, location_probability, dynamic_score),
                    1,
                    1,
                    timestamp,
                )
                track.samples.append(sample)
                self.tracks.append(track)
                matched_tracks.add(id(track))
                continue
            matched_tracks.add(id(track))
            elapsed = timestamp - track.last_time
            if elapsed < self.minimum_interval:
                continue
            measured_velocity = (position - track.position) / max(elapsed, 1e-6)
            speed = float(np.linalg.norm(measured_velocity))
            if speed > 3.0:
                measured_velocity *= 3.0 / speed
            track.velocity = 0.75 * track.velocity + 0.25 * measured_velocity
            support = self.evidence_agent.support(probability, location_probability, dynamic_score)
            weight = min(0.45, max(0.15, support))
            track.position = (1.0 - weight) * track.position + weight * position
            track.evidence += support
            track.hits += 1
            track.consecutive_hits = track.consecutive_hits + 1 if elapsed <= 0.60 else 1
            track.last_time = timestamp
            track.samples.append(DetectionSample(position.copy(), observer.copy(), probability, location_probability, dynamic_score, timestamp))
            static_source_distance = min(
                (float(np.linalg.norm(track.position - source)) for source in static_source_positions),
                default=float("inf"),
            )
            mode = self.confirmation_agent.mode(track, self, static_source_distance <= 4.0)
            if mode is not None and not self._is_suppressed(track.position, estimated_count):
                confirmed.append((track.evidence * location_probability, track, mode))
        if not accepted:
            self._reset_continuity()
        if not confirmed:
            return None
        _, track, mode = max(confirmed, key=lambda item: item[0])
        mean_probability = float(np.mean([sample.probability for sample in track.samples]))
        track.emitted = True
        self.emitted_positions.append(track.position.copy())
        return PingDecision(track.position.copy(), mean_probability, mode)
