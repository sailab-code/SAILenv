#
# Copyright (C) 2020 Enrico Meloni, Luca Pasqualini
# University of Siena - Artificial Intelligence Laboratory - SAILab
#
#
# L2S is licensed under a MIT license.
#
# You should have received a copy of the license along with this
# work. If not, see <https://en.wikipedia.org/wiki/MIT_License>.

# Import packages

import base64
import numpy as np

# Import src

from client.agent_api import AgentApi


class Agent:
    """
    TODO: summary ??? Maybe some more check to avoid connection errors?

    """
    def __init__(self,
                 main_frame_active: bool = True,
                 object_frame_active: bool = True,
                 category_frame_active: bool = True,
                 flow_frame_active: bool = True):
        self.flow_frame_active: bool = flow_frame_active
        self.category_frame_active: bool = category_frame_active
        self.object_frame_active: bool = object_frame_active
        self.main_frame_active: bool = main_frame_active
        self.api: AgentApi = AgentApi()
        self.id: int = -1

    def register(self):
        """
        Register the agent on the Unity server and set its id.
        """
        self.id = self.api.register()["Content"]

    def delete(self):
        """
        Delete the agent on the Unity server.
        """
        self.api.delete(self.id)

    def get_frame(self):
        """
        Get the frame from the cameras on the Unity server.

        :return: a dict of frames indexed by keywords main, object, category and flow.
        """
        response = self.api.get_frame(self.id,
                                      self.main_frame_active,
                                      self.object_frame_active,
                                      self.category_frame_active,
                                      self.flow_frame_active)

        if response["Error"] is True:
            return None

        base64_images = response["Content"]
        frame = {}

        if self.main_frame_active:
            frame["main"] = self.__decode_image(base64_images["Main"])
        else:
            frame["main"] = None

        if self.object_frame_active:
            frame["object"] = self.__decode_image(base64_images["Object"])
        else:
            frame["object"] = None

        if self.category_frame_active:
            frame["category"] = self.__decode_image(base64_images["Category"])
        else:
            frame["category"] = None

        if self.flow_frame_active:
            frame["flow"] = self.__decode_image(base64_images["Optical"])
        else:
            frame["flow"] = None

        return frame

    @staticmethod
    def __decode_image(b64_input) -> np.ndarray:
        """
        Decode an image from the given base64 representation to a numpy array.

        :param b64_input: the base64 representation of an image
        :return: the numpy array representation of an image
        """
        return np.frombuffer(base64.b64decode(b64_input), np.uint8)
