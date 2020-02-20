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

import time
import numpy as np
try:
    from cv2 import cv2
except ImportError:
    import cv2

# Import src

from agent import Agent

frames: int = 1000


def decode_image(array: np.ndarray):
    """
    Decode the given numpy array with OpenCV.
    :param array: the numpy array to decode
    :return: the decode image that can be displayed
    """
    img = cv2.imdecode(array, cv2.IMREAD_COLOR)
    return img


if __name__ == '__main__':
    print("Generating agent...")
    agent = Agent(flow_frame_active=False)
    print("Registering agent on server...")
    agent.register()
    print(f"Agent registered with ID: {agent.id}")
    last_unity_time: float = 0.0
    try:
        print("Press ESC to close")
        while True:
            start_real_time = time.time()
            start_unity_time = last_unity_time
            frame = agent.get_frame()

            if frame["main"] is not None:
                main_img = decode_image(frame["main"])
                cv2.imshow("PBR", main_img)

            if frame["category"] is not None:
                cat_img = decode_image(frame["category"])
                cv2.imshow("Category ID", cat_img)

            if frame["object"] is not None:
                obj_img = decode_image(frame["object"])
                cv2.imshow("Object ID", obj_img)

            if frame["flow"] is not None:
                flow_img = decode_image(frame["flow"])
                cv2.imshow("Optical Flow", flow_img)

            key = cv2.waitKey(1)
            print(f"Real elapsed time: {time.time() - start_real_time}")
            if key == 27:  # ESC Pressed
                break
    finally:
        print(f"Closing agent {agent.id}")
        agent.dispose()
