from dataclasses import dataclass, field, fields
from typing import List, Optional, Dict

from sailenv.generators.object import Object
from sailenv.generators.timings import Timings, AllTogetherTimings
from sailenv.utilities import BaseDataclass


@dataclass
class Scenario(BaseDataclass):
    scene_name: str
    objects: List[Object]
    timings: Timings = field(default_factory=AllTogetherTimings)
    spawn_frustum: bool = False

    @property
    def n_objects(self):
        return len(self.objects)