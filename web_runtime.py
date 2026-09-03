from __future__ import annotations

import numpy as np

from config import BOARD_HEIGHT, BOARD_WIDTH, ENVIRONMENT_PRESETS, PERCEPTION_CELL_SIZE, SENSOR_RANGE
from integration.controlled_stepper import ControlledBoardStepper
from integration.coverage_board_runtime import CoverageBoardRuntime
from integration.human_detector_runtime import HumanDetectorResult, HumanDetectorRuntime
from integration.rescue_manager import RescueManager
from simulation.board import Board


class WebSimulationSession:
    def __init__(self) -> None:
        self.seed = 42
        self.environment = next(iter(ENVIRONMENT_PRESETS))
        self.running = False
        self.navigation_mode = "CNN Agent"
        self.manual_action = "forward"
        self.speed = 4
        self.noise_level = 0.0
        self.robot_perception = False
        self.frame = 0
        self.force_full_state = True
        self.last_action = "stop"
        self.coverage_ratio = 0.0
        self.detector_error: str | None = None
        self.coverage_error: str | None = None
        self.detector_runtime: HumanDetectorRuntime | None = None
        self.coverage_runtime: CoverageBoardRuntime | None = None
        try:
            self.detector_runtime = HumanDetectorRuntime()
        except Exception as error:
            self.detector_error = str(error)
        try:
            self.coverage_runtime = CoverageBoardRuntime()
        except Exception as error:
            self.coverage_error = str(error)
        self.stepper = ControlledBoardStepper()
        self.rescue_manager = RescueManager()
        self.board = Board(self.seed, self.environment, self.noise_level)
        self.detector_result = HumanDetectorResult(0.0, 0, "WARMING UP", 0.0, None, 0, 0)
        self._update_detector()

    def reset(self, seed: int | None = None, environment: str | None = None) -> None:
        if seed is not None:
            self.seed = seed
        if environment is not None:
            self.environment = environment
        self.board = Board(self.seed, self.environment, self.noise_level)
        self.running = False
        self.last_action = "stop"
        self.coverage_ratio = 0.0
        self.frame = 0
        self.force_full_state = True
        self.stepper.reset()
        self.rescue_manager.reset()
        if self.detector_runtime is not None:
            self.detector_runtime.reset()
        if self.coverage_runtime is not None:
            self.coverage_runtime.reset()
        self.detector_result = HumanDetectorResult(0.0, 0, "WARMING UP", 0.0, None, 0, 0)
        self._update_detector()

    def apply_command(self, command: dict) -> None:
        command_type = str(command.get("type", ""))
        value = command.get("value")
        if command_type == "run":
            self.running = True
        elif command_type == "stop":
            self.running = False
            self.last_action = "stop"
        elif command_type == "reset":
            self.reset()
        elif command_type == "randomize":
            self.reset(self.seed + 1)
        elif command_type == "environment" and value in ENVIRONMENT_PRESETS:
            self.reset(self.seed, str(value))
        elif command_type == "navigation_mode" and value in ("Manual", "CNN Agent"):
            self.navigation_mode = str(value)
        elif command_type == "manual_action" and value in ("forward", "turn_left", "turn_right", "stop"):
            self.manual_action = str(value)
        elif command_type == "speed":
            self.speed = int(np.clip(int(value), 1, 10))
        elif command_type == "noise_level":
            self.noise_level = float(np.clip(float(value), 0.0, 5.0))
            self.board.set_enhanced_noise_level(self.noise_level)
        elif command_type == "robot_perception":
            self.robot_perception = bool(value)
            self.force_full_state = True

    def _update_detector(self) -> None:
        if self.detector_runtime is None:
            return
        try:
            result = self.detector_runtime.process_board(self.board)
            self.detector_result = result
            rescue_event = self.rescue_manager.process(self.board, result)
            if rescue_event is not None:
                self.detector_runtime.ignore_position(rescue_event.position)
        except Exception as error:
            self.detector_error = str(error)
            self.detector_runtime = None

    def step(self) -> None:
        if self.rescue_manager.all_rescued(self.board):
            self.running = False
            self.last_action = "stop"
            return
        if self.navigation_mode == "CNN Agent" and self.coverage_runtime is not None:
            try:
                action, coverage = self.coverage_runtime.decide(self.board)
                self.coverage_ratio = coverage
            except Exception as error:
                self.coverage_error = str(error)
                self.coverage_runtime = None
                action = "forward"
        else:
            action = self.manual_action
            self.coverage_ratio = float(self.board.perception.observed.mean())
        self.last_action = action
        self.stepper.step(self.board, action, 0.4)
        self._update_detector()
        self.frame += 1

    def tick_delay(self) -> float:
        return max(0.08, 0.40 / (0.7 + 0.5 * self.speed))

    @staticmethod
    def _points(values, digits: int = 2) -> list[list[float]]:
        return [[round(float(item[0]), digits), round(float(item[1]), digits)] for item in values]

    def _perception_state(self) -> dict:
        perception = self.board.perception
        anomaly = perception.magnetic_anomaly()
        rows, columns = np.where(perception.observed)
        cells = []
        for row, column in zip(rows, columns):
            value = anomaly[row, column]
            cells.append([int(column), int(row), None if not np.isfinite(value) else round(float(value), 2)])
        return {
            "cells": cells,
            "rows": perception.rows,
            "columns": perception.columns,
            "cell_size": PERCEPTION_CELL_SIZE,
            "obstacle_points": self._points(perception.obstacle_points[-900::2]),
            "metal_candidates": self._points(perception.metal_candidates()),
            "range_endpoints": self._points(self.board.range_scan.endpoints[::3]),
        }

    def _world_state(self) -> dict:
        if self.robot_perception:
            return {"perception": self._perception_state()}
        return {
            "obstacles": [
                {"x": round(float(item.x), 2), "y": round(float(item.y), 2), "width": round(float(item.width), 2), "height": round(float(item.height), 2)}
                for item in self.board.obstacles
            ],
            "metals": [
                {"x": round(float(item.position[0]), 2), "y": round(float(item.position[1]), 2), "size": round(float(item.size), 2)}
                for item in self.board.metals
            ],
            "people": [
                {
                    "x": round(float(item.position[0]), 2),
                    "y": round(float(item.position[1]), 2),
                    "rescued": index in self.rescue_manager.rescued_indices,
                }
                for index, item in enumerate(self.board.people)
            ],
        }

    def snapshot(self, include_map: bool = True) -> dict:
        board = self.board
        measurement = board.measurements[-1]
        result = self.detector_result
        markers = [
            {
                "x": round(float(marker.position[0]), 2),
                "y": round(float(marker.position[1]), 2),
                "confidence": round(float(marker.confidence), 4),
                "timestamp": round(float(marker.timestamp), 2),
                "observations": int(marker.observations),
            }
            for marker in board.detection_tracker.markers
        ]
        history = board.measurements[-120:]
        state = {
            "type": "state",
            "frame": self.frame,
            "running": self.running,
            "status": "complete" if self.rescue_manager.all_rescued(board) else ("running" if self.running else "stopped"),
            "time": round(float(board.time), 2),
            "seed": self.seed,
            "environment": self.environment,
            "environments": list(ENVIRONMENT_PRESETS),
            "robot_perception": self.robot_perception,
            "navigation_mode": self.navigation_mode,
            "manual_action": self.manual_action,
            "speed": self.speed,
            "noise_level": round(self.noise_level, 1),
            "board": {"width": BOARD_WIDTH, "height": BOARD_HEIGHT, "sensor_range": SENSOR_RANGE},
            "robot": {
                "x": round(float(board.robot.position[0]), 3),
                "y": round(float(board.robot.position[1]), 3),
                "heading": round(float(board.robot.heading), 4),
                "moving": bool(board.robot.moving),
                "path": self._points(board.robot.path),
            },
            "sensor": {
                "bx": round(float(measurement.bx), 4),
                "by": round(float(measurement.by), 4),
                "bz": round(float(measurement.bz), 4),
                "magnitude": round(float(measurement.magnitude), 4),
                "history": [
                    [round(float(item.timestamp), 2), round(float(item.bx), 3), round(float(item.by), 3), round(float(item.bz), 3), round(float(item.magnitude), 3)]
                    for item in history
                ],
            },
            "detector": {
                "probability": round(float(result.person_probability), 5),
                "predicted_label": int(result.predicted_label),
                "status": result.status if self.detector_runtime is not None else "ERROR",
                "window_size": int(result.window_size),
                "estimated_count": int(result.estimated_count),
                "model": "best_human_transition_detector.pth",
                "error": self.detector_error,
            },
            "agent": {
                "action": self.last_action,
                "coverage": round(float(self.coverage_ratio), 5),
                "state": "COMPLETE" if self.rescue_manager.all_rescued(board) else ("PATROLLING" if self.coverage_runtime is not None and self.coverage_runtime.patrol_mode else "EXPLORING"),
                "model": "best_coverage_agent.pth",
                "error": self.coverage_error,
            },
            "markers": markers,
            "marker_count": len(markers),
            "rescued_count": None if self.robot_perception else len(self.rescue_manager.rescued_indices),
            "actual_people": None if self.robot_perception else len(board.people),
        }
        if include_map or self.force_full_state:
            state.update(self._world_state())
            self.force_full_state = False
        return state

