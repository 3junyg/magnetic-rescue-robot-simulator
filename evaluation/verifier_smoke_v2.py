import ast
from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from integration.candidate_verifier_runtime_v2 import CandidateVerifierRuntime
from simulation.observations_v2 import observation_from_board
from ui.renderer_v2 import render_board
from web_runtime_v2 import WebSimulationSession


def main():
    root = Path(__file__).resolve().parents[1]
    files = list(root.rglob('*.py'))
    for path in files:
        ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
    print(f'syntax_checked={len(files)}')
    for relative in ('ml/candidate_patch_v2.py', 'integration/candidate_verifier_runtime_v2.py'):
        source = (root / relative).read_text(encoding='utf-8')
        for name in ('people', 'metals', 'ground_truth', 'human_field', 'metal_field'):
            assert name not in source, (relative, name)
    session = WebSimulationSession()
    assert session.detector_error is None, session.detector_error
    assert session.detector_runtime.verifier_enabled
    verifier = session.detector_runtime.verifier
    assert verifier is not None
    session.robot_perception = True
    started = time.perf_counter()
    for _ in range(100):
        session.step()
        assert session.detector_error is None, session.detector_error
    elapsed = time.perf_counter() - started
    observation = observation_from_board(session.board)
    size = len(verifier.history.frames)
    verifier.observe(observation)
    assert len(verifier.history.frames) == size
    scores = verifier.scores([observation.robot_position])
    assert scores.shape == (1,) and np.isfinite(scores).all()
    for perception in (False, True):
        figure = render_board(session.board, perception)
        plt.close(figure)
    session.reset()
    assert len(verifier.history.frames) == 1
    assert verifier.rejected_candidates == 0
    assert not session.board.detection_tracker.markers
    session.apply_command({'type': 'randomize'})
    for _ in range(40):
        session.step()
    assert session.detector_error is None
    verifier.reset()
    assert not verifier.history.frames
    assert verifier.rejected_candidates == 0
    assert verifier.last_probability == 0.0
    try:
        CandidateVerifierRuntime(root / 'models' / 'missing_verifier_v2.pth')
    except FileNotFoundError as error:
        assert 'Candidate verifier model not found' in str(error)
    else:
        raise AssertionError('Missing model did not raise an error')
    print(f'observation_only=PASS headless_100=PASS perception_OFF_ON=PASS reset=PASS randomize=PASS missing_model=PASS elapsed_100={elapsed:.2f}s')


if __name__ == '__main__':
    main()
