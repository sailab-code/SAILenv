import json
from dataclasses import dataclass, asdict

from sailenv.utilities import BaseDataclass


@dataclass
class Dynamic(BaseDataclass):
    def todict(self):
        return {
            "type": self.get_type(),
            "description": self.asdict()
        }

    @staticmethod
    def get_type():
        raise NotImplemented()