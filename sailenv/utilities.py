#
# Copyright (C) 2020 Enrico Meloni, Luca Pasqualini, Matteo Tiezzi
# University of Siena - Artificial Intelligence Laboratory - SAILab
#
#
# SAILenv is licensed under a MIT license.
#
# You should have received a copy of the license along with this
# work. If not, see <https://en.wikipedia.org/wiki/MIT_License>.

# Import packages
import copy
from dataclasses import dataclass, asdict, fields, _is_dataclass_instance

import cv2
import numpy as np


def draw_flow_map(optical_flow):
    hsv = np.zeros((optical_flow.shape[0], optical_flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 1] = 255

    mag, ang = cv2.cartToPolar(optical_flow[..., 0], optical_flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    frame_flow_map = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return frame_flow_map


@dataclass
class BaseDataclass:
    @classmethod
    def __asdict_inner(self, obj):
        if _is_dataclass_instance(obj):
            result = []
            for f in fields(obj):
                val = getattr(obj, f.name)
                if hasattr(val, 'todict'):
                    result.append((f.name, val.todict()))
                else:
                    result.append((f.name, self.__asdict_inner(val)))
            return dict(result)
        elif isinstance(obj, tuple) and hasattr(obj, '_fields'):
            return type(obj)(*[self.__asdict_inner(v) for v in obj])
        elif isinstance(obj, (list, tuple)):
            return type(obj)(self.__asdict_inner(v) for v in obj)
        elif isinstance(obj, dict):
            return type(obj)((self.__asdict_inner(k),
                              self.__asdict_inner(v))
                             for k, v in obj.items())
        else:
            return copy.deepcopy(obj)

    def todict(self):
        return self.__asdict_inner(self)

    def asdict(self):
        return self.__asdict_inner(self)

