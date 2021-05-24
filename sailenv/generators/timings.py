from dataclasses import dataclass, field
from typing import Dict

from sailenv.utilities import BaseDataclass


@dataclass
class Timings(BaseDataclass):
    def todict(self):
        return {
            "type": self.get_type(),
            "description": self.asdict()
        }

    @staticmethod
    def get_type():
        raise NotImplemented()


@dataclass
class AllTogetherTimings(Timings):
    wait_time: float

    @staticmethod
    def get_type():
        return "all_together"


@dataclass
class WaitUntilCompleteTimings(Timings):
    wait_time: float

    @staticmethod
    def get_type():
        return "wait_until_complete"


@dataclass
class DictTimings(Timings):
    times: Dict[int, float] = field(default_factory=dict)

    @staticmethod
    def get_type():
        return "dict"