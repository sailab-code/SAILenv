from dataclasses import dataclass

from sailenv import Vector3
from sailenv.dynamics import Dynamic


@dataclass
class UniformMovementRandomBounce(Dynamic):
    start_direction: Vector3 = Vector3(0, 0, 1)
    speed: float = 5
    angular_speed: float = 2
    seed: int = 42

    @staticmethod
    def get_type():
        return "uniform_movement_random_bounce"