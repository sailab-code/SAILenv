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

import base64
import numpy as np
from old_http.agent_api import AgentApi
from old_http.simulation_api import SimulationApi
import cv2


# Import src

class Agent:
    """
    TODO: summary ??? Maybe some more check to avoid connection errors?

    """
    def __init__(self,
                 main_frame_active: bool = True,
                 object_frame_active: bool = True,
                 category_frame_active: bool = True,
                 flow_frame_active: bool = True,
                 depth_frame_active: bool = True,
                 host: str = "localhost",
                 port: int = 8080,
                 width: int = 512,
                 height: int = 384):
        """

        :param main_frame_active: True if the virtual world should generate the main camera view
        :param object_frame_active: True if the virtual world should generate object instance supervisions
        :param category_frame_active: True if the virtual world should generate category supervisions
        :param flow_frame_active: True if the virtual world should generate optical flow data
        :param host: address on which the unity virtual world is listening
        :param port: port on which the unity virtual world is listening
        :param width: width of the stream, should be multiple of 8
        :param height: height of the stream, should be multiple of 8
        """
        self.flow_frame_active: bool = flow_frame_active
        self.category_frame_active: bool = category_frame_active
        self.object_frame_active: bool = object_frame_active
        self.main_frame_active: bool = main_frame_active
        self.depth_frame_active: bool = depth_frame_active
        self.api: AgentApi = AgentApi(host=host, port=port)
        self.sim_api: SimulationApi = SimulationApi(host=host, port=port)
        self.id: int = -1
        self.width = width
        self.height = height

    def register(self):
        """
        Register the agent on the Unity server and set its id.
        """
        self.id = self.api.register(width=self.width, height=self.height)["Content"]

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
        response, total_seconds = self.api.get_frame(self.id,
                                      self.main_frame_active,
                                      self.object_frame_active,
                                      self.category_frame_active,
                                      self.flow_frame_active,
                                      self.depth_frame_active)

        if response["Error"] is True:
            return None

        content = response["Content"]
        frame = {}

        if self.main_frame_active:
            frame["main"] = cv2.imdecode(self.__decode_image(content["Main"]), cv2.IMREAD_COLOR)
        else:
            frame["main"] = None

        if self.object_frame_active:
            frame["object"] = self.__decode_image(content["Object"])  # TODO
        else:
            frame["object"] = None

        if self.category_frame_active:
            cat_frame = self.__decode_category(content["Category"])
            cat_frame = np.reshape(cat_frame, (self.height, self.width, 3))
            cat_frame = cat_frame[:,:,0]
            frame["category"] = cat_frame.flatten()
            #frame["category_debug"] = self.__decode_image(base64_images["CategoryDebug"])
        else:
            frame["category"] = None
            #frame["category_debug"] = None

        if self.flow_frame_active:
            flow = self.__decode_image(content["Optical"], np.float32)  # TODO
            frame["flow"] = self.__decode_flow(flow)
        else:
            frame["flow"] = None

        if self.depth_frame_active:
            frame["depth"] = cv2.imdecode(self.__decode_image(content["Depth"]), cv2.IMREAD_COLOR)
        else:
            frame["depth"] = None

        if "Timings" in content:
            # Convert elapsed milliseconds to seconds
            frame["timings"] = {key: (value / 1000) for key, value in content["Timings"].items()}
            frame["timings"]["Http"] = total_seconds

        return frame

    def get_resolution(self):
        """

        :return: (height, width, channels, fps)
        """

        content = self.sim_api.get_resolution()
        return content["Height"], content["Width"], 3, 25

    def get_categories(self):
        content = self.sim_api.get_categories()
        categories = {}
        for cat in content:
            categories[cat["Name"]] = cat["Id"]

        return categories

    @staticmethod
    def __decode_image(b64_input, type=np.uint8) -> np.ndarray:
        """
        Decode an image from the given base64 representation to a numpy array.

        :param b64_input: the base64 representation of an image
        :return: the numpy array representation of an image
        """
        return np.frombuffer(base64.b64decode(b64_input), type)

    def __decode_category(self, b64_input) -> np.ndarray:
        """
        Decode the category supervisions from the given base64 representation to a numpy array.
        :param b64_input: the base64 representation of categories
        :return: the numpy array containing the category supervisions
        """

        cat_frame = Agent.__decode_image(b64_input)
        cat_frame = np.reshape(cat_frame, (self.height, self.width, -1))
        cat_frame = np.flipud(cat_frame)
        cat_frame = np.reshape(cat_frame, (-1))
        cat_frame = np.ascontiguousarray(cat_frame)
        return cat_frame

    def __decode_flow(self, flow_frame: np.ndarray) -> np.ndarray:
        flow = flow_frame
        flow = np.reshape(flow, (self.height, self.width, -1))
        flow = np.flipud(flow)
        flow = np.ascontiguousarray(flow)
        return flow
