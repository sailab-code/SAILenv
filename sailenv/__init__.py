from typing import Tuple
from dataclasses import dataclass

@dataclass
class Vector3:
    x: float
    y: float
    z: float

    def __getitem__(self, item):
        if item == 0:
            return self.x
        elif item == 1:
            return self.y
        elif item == 2:
            return self.z
        else:
            raise ValueError("index must be between 0 and 2")

    @staticmethod
    def from_tuple(tuple: Tuple[float, float, float]):
        return Vector3(*tuple)