import argparse
from io import BytesIO
from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from web_runtime_v2 import WebSimulationSession
from ui.renderer_v2 import render_board, render_sensor_history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dpi', type=int, default=200)
    args = parser.parse_args()
    traces, timings, rendered = [], [], []
    for seed in (11, 73, 99):
        session = WebSimulationSession()
        started = time.perf_counter()
        trace = []
        for _ in range(500):
            session.step()
            result = session.detector_result
            estimate = result.estimated_position
            trace.append([*session.board.robot.position, result.person_probability, *(estimate if estimate is not None else [np.nan, np.nan]), len(session.rescue_manager.rescued_indices)])
        timings.append(time.perf_counter() - started)
        traces.append(trace)
        started = time.perf_counter()
        for _ in range(3):
            for figure in (render_board(session.board, True), render_sensor_history(session.board)):
                figure.savefig(BytesIO(), format='png', dpi=args.dpi, bbox_inches='tight')
                plt.close(figure)
        rendered.append((time.perf_counter() - started) / 3)
        print(f'seed={seed} steps_500={timings[-1]:.3f}s render_pair={rendered[-1]:.3f}s', flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, traces=traces, step_seconds=timings, render_seconds=rendered, dpi=args.dpi)


if __name__ == '__main__':
    main()
