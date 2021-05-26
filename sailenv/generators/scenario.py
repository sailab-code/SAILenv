from dataclasses import dataclass, field, fields
from typing import List, Optional, Dict

from sailenv.generators.object import Object
from sailenv.generators.timings import Timings, AllTogetherTimings
from sailenv.utilities import BaseDataclass


@dataclass
class Frustum(BaseDataclass):
    spawn: bool = True
    far_clip_plane: float = 15.

    @staticmethod
    def default_frustum():
        return Frustum(False, -1)


@dataclass
class Scenario(BaseDataclass):
    scene_name: str
    objects: List[Object]
    timings: Timings = field(default_factory=AllTogetherTimings)
    frustum: Frustum = field(default_factory=Frustum.default_frustum)

    @property
    def n_objects(self):
        return len(self.objects)