import argparse

import numpy as np

from config_v2 import ENVIRONMENT_PRESETS
from web_runtime_v2 import WebSimulationSession


def delta_peak(previous, current, robot_position):
    old = {tuple(point): vector for point, vector in zip(previous.positions, previous.vectors)}
    points = []
    differences = []
    for point, vector in zip(current.positions, current.vectors):
        prior = old.get(tuple(point))
        if prior is not None:
            points.append(point)
            differences.append(vector - prior)
    if not points:
        return None
    points = np.asarray(points)
    differences = np.asarray(differences)
    differences -= np.median(differences, axis=0)
    distances = np.linalg.norm(points - robot_position, axis=1)
    active = distances <= 8.0
    if not active.any():
        return None
    norms = np.linalg.norm(differences[active], axis=1)
    index = int(np.argmax(norms))
    return points[active][index], float(norms[index])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, nargs='+', default=[1, 3, 42])
    parser.add_argument('--steps', type=int, default=400)
    args = parser.parse_args()
    records = []
    for environment in ENVIRONMENT_PRESETS:
        for seed in args.seeds:
            session = WebSimulationSession()
            session.reset(seed, environment)
            for _ in range(args.steps):
                previous = session.board.magnetic_scan
                session.step()
                if session.detector_result.status != 'READY':
                    continue
                peak = delta_peak(previous, session.board.magnetic_scan, session.board.robot.position)
                if peak is None:
                    continue
                point, magnitude = peak
                person_distance = min((np.linalg.norm(person.position - point) for person in session.board.people if person.signal_strength > 0), default=float('inf'))
                robot_person_distance = min((np.linalg.norm(person.position - session.board.robot.position) for person in session.board.people if person.signal_strength > 0), default=float('inf'))
                records.append((magnitude, person_distance, robot_person_distance, session.detector_result.person_probability))
    for threshold in (1.5, 2.0, 3.0, 4.0, 6.0):
        selected = [record for record in records if record[0] >= threshold]
        good = sum(record[1] <= 6.5 for record in selected)
        print(f'threshold={threshold:.1f} peaks={len(selected)} near_person={good} precision={good / max(1, len(selected)):.3f}')
    print(f'total={len(records)}')


if __name__ == '__main__':
    main()
