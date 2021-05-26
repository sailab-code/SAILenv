from dataclasses import dataclass, asdict
from typing import List

from sailenv import Vector3
from sailenv.dynamics import Dynamic
from sailenv.utilities import BaseDataclass


@dataclass
class Object(BaseDataclass):
    id: str
    prefab: str
    position: Vector3
    rotation: Vector3
    dynamic: Dynamic = None
    frustum_limited: bool = True

    def asdict(self):
        return asdict(self)