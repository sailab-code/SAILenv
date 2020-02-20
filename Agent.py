import base64

import numpy as np

from api.AgentApi import AgentApi


class Agent:
    def __init__(self,
                 main_frame_active: bool = True,
                 object_frame_active: bool = True,
                 category_frame_active: bool = True,
                 flow_frame_active: bool = True
    ):
        self.flow_frame_active = flow_frame_active
        self.category_frame_active = category_frame_active
        self.object_frame_active = object_frame_active
        self.main_frame_active = main_frame_active

        self.api = AgentApi()
        self.id = -1
        return

    def register(self):
        self.id = self.api.register()["Content"]

    @staticmethod
    def __decode_image(b64_input):
        img = np.frombuffer(base64.b64decode(b64_input), np.uint8)
        return img

    def dispose(self):
        self.api.delete(self.id)
        return

    def get_frame(self):
        response = self.api.get_frame(
            self.id,
            self.main_frame_active,
            self.object_frame_active,
            self.category_frame_active,
            self.flow_frame_active
        )

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
            frame["flow"] = self.__decode_image(base64_images["Flow"])
        else:
            frame["flow"] = None

        return frame
