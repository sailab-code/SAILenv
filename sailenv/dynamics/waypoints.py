from dataclasses import dataclass, field
from typing import List

from sailenv import Vector3
from sailenv.dynamics import Dynamic
from sailenv.utilities import BaseDataclass


@dataclass
class Waypoint(BaseDataclass):
    position: Vector3 = Vector3(0, 0, 0)
    rotation: Vector3 = Vector3(0, 0, 0)


@dataclass
class WaypointsDynamic(Dynamic):
    waypoints: List[Waypoint] = field(default_factory=list)
    total_time: float = 10.0


@dataclass
class LinearWaypoints(WaypointsDynamic):
    @staticmethod
    def get_type():
        return "linear_waypoints"


@dataclass
class CatmullWaypoints(WaypointsDynamic):
    @staticmethod
    def get_type():
        return "catmull_waypoints"