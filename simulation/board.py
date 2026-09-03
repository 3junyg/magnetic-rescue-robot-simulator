import numpy as np

from config import BOARD_HEIGHT, BOARD_WIDTH, ENVIRONMENT_PRESETS, HISTORY_LENGTH, METAL_STRENGTH_MAX, METAL_STRENGTH_MIN, PERSON_MAX_SPEED, PERSON_MIN_SPEED, PERSON_SIGNAL_MAX, PERSON_SIGNAL_MIN, ROBOT_RADIUS, ROBOT_SPEED
from simulation.detection_tracker import DetectionTracker
from simulation.enhanced_noise import EnhancedNoiseSensor
from simulation.entities import Metal, Obstacle, Person, Robot
from simulation.magnetic_field import MagneticFieldSensor
from simulation.perception import RobotPerception
from simulation.range_sensor import RangeSensor


class Board:
    def __init__(self, seed: int, environment: str = "도심 침수 지역", enhanced_noise_level: float = 0.0) -> None:
        self.seed = seed
        self.environment = environment
        self.settings = ENVIRONMENT_PRESETS[environment]
        self.rng = np.random.default_rng(seed)
        self.time = 0.0
        self.obstacles = self._create_obstacles()
        self.people = self._create_people()
        self.metals = self._create_metals()
        self.robot = Robot(self._free_point(3.0), self.rng.uniform(-np.pi, np.pi), ROBOT_RADIUS, ROBOT_SPEED)
        self.robot.path.append(self.robot.position.copy())
        self.sensor = EnhancedNoiseSensor(MagneticFieldSensor(self.settings["noise"], self.rng), seed, enhanced_noise_level)
        self.measurements = [self.sensor.measure(self)]
        self.magnetic_scan = self.sensor.scan(self)
        self.range_sensor = RangeSensor()
        self.range_scan = self.range_sensor.scan(self.robot.position, self.obstacles)
        self.perception = RobotPerception()
        self.perception.update(self.robot.position, self.magnetic_scan, self.range_scan)
        self.detection_tracker = DetectionTracker()

    def set_enhanced_noise_level(self, level: float) -> None:
        self.sensor.set_level(level)

    def _create_obstacles(self) -> list[Obstacle]:
        obstacles = []
        for _ in range(self.settings["obstacles"]):
            vertical = self.rng.random() < 0.5
            width = self.rng.uniform(2.5, 5.0) if vertical else self.rng.uniform(7.0, 15.0)
            height = self.rng.uniform(7.0, 15.0) if vertical else self.rng.uniform(2.5, 5.0)
            x = self.rng.uniform(2.0, BOARD_WIDTH - width - 2.0)
            y = self.rng.uniform(2.0, BOARD_HEIGHT - height - 2.0)
            obstacles.append(Obstacle(x, y, width, height))
        return obstacles

    def _is_free(self, point: np.ndarray, margin: float) -> bool:
        if not (margin <= point[0] <= BOARD_WIDTH - margin):
            return False
        if not (margin <= point[1] <= BOARD_HEIGHT - margin):
            return False
        return not any(obstacle.contains(point, margin) for obstacle in self.obstacles)

    def _free_point(self, margin: float) -> np.ndarray:
        for _ in range(5000):
            point = np.array([self.rng.uniform(margin, BOARD_WIDTH - margin), self.rng.uniform(margin, BOARD_HEIGHT - margin)])
            if self._is_free(point, margin):
                return point
        raise RuntimeError("빈 위치를 생성할 수 없습니다.")

    def _create_people(self) -> list[Person]:
        people = []
        for _ in range(self.settings["people"]):
            position = self._free_point(1.0)
            heading = self.rng.uniform(-np.pi, np.pi)
            speed = self.rng.uniform(PERSON_MIN_SPEED, PERSON_MAX_SPEED)
            velocity = speed * np.array([np.cos(heading), np.sin(heading)])
            people.append(Person(position, velocity, heading, self.rng.uniform(PERSON_SIGNAL_MIN, PERSON_SIGNAL_MAX), self.rng.uniform(0.0, 2.0 * np.pi, 3)))
        return people

    def _create_metals(self) -> list[Metal]:
        return [Metal(self._free_point(1.0), self.rng.uniform(1.2, 4.0), self.rng.uniform(METAL_STRENGTH_MIN, METAL_STRENGTH_MAX), self.rng.uniform(-np.pi, np.pi)) for _ in range(self.settings["metals"])]

    def _update_people(self, dt: float) -> None:
        for person in self.people:
            person.heading += self.rng.normal(0.0, 0.22) * dt
            speed = float(np.linalg.norm(person.velocity))
            velocity = speed * np.array([np.cos(person.heading), np.sin(person.heading)])
            candidate = person.position + velocity * dt
            if not self._is_free(candidate, 0.65):
                person.heading += self.rng.uniform(0.65 * np.pi, 1.35 * np.pi)
                velocity = speed * np.array([np.cos(person.heading), np.sin(person.heading)])
                candidate = person.position + velocity * dt
            if self._is_free(candidate, 0.65):
                person.position = candidate
            person.velocity = velocity
            person.variation = 0.8 * person.variation + self.rng.normal(0.0, 0.30)

    def _update_robot(self, dt: float) -> None:
        if self.robot.turn_remaining > 0.0:
            turn = min(1.8 * dt, self.robot.turn_remaining)
            self.robot.heading += turn
            self.robot.turn_remaining -= turn
        direction = np.array([np.cos(self.robot.heading), np.sin(self.robot.heading)])
        look_ahead = self.robot.position + direction * max(3.0, self.robot.speed * dt * 3.0)
        candidate = self.robot.position + direction * self.robot.speed * dt
        if not self._is_free(look_ahead, self.robot.radius) or not self._is_free(candidate, self.robot.radius):
            self.robot.turn_remaining = self.rng.uniform(0.45 * np.pi, 0.9 * np.pi)
            self.robot.heading += 1.8 * dt
        elif self._is_free(candidate, self.robot.radius):
            self.robot.position = candidate
        self.robot.heading = (self.robot.heading + np.pi) % (2.0 * np.pi) - np.pi
        self.robot.path.append(self.robot.position.copy())
        if len(self.robot.path) > HISTORY_LENGTH:
            self.robot.path.pop(0)

    def step(self, dt: float, update_perception: bool = True) -> None:
        self.time += dt
        self._update_people(dt)
        self._update_robot(dt)
        measurement = self.sensor.measure(self)
        self.measurements.append(measurement)
        self.magnetic_scan = self.sensor.scan(self)
        if update_perception:
            self.range_scan = self.range_sensor.scan(self.robot.position, self.obstacles)
            self.perception.update(self.robot.position, self.magnetic_scan, self.range_scan)
        if len(self.measurements) > HISTORY_LENGTH:
            self.measurements.pop(0)
