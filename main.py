import base64
import time

import numpy as np

from Agent import Agent
from api.CameraApi import CameraApi
import cv2

try:
    from cv2 import cv2
except ImportError:
    pass

frames = 1000


def decode_image(np_arr):
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return img

if __name__ == '__main__':
    agent = Agent(flow_frame_active=False)
    agent.register()
    last_unity_time = 0

    try:
        while True:
            start_real_time = time.time()
            start_unity_time = last_unity_time
            frame = agent.get_frame()

            if frame["main"] is not None:
                main_img = decode_image(frame["main"])
                cv2.imshow("main", main_img)

            if frame["category"] is not None:
                cat_img = decode_image(frame["category"])
                cv2.imshow("cat", cat_img)

            if frame["object"] is not None:
                obj_img = decode_image(frame["object"])
                cv2.imshow("object", obj_img)

            if frame["flow"] is not None:
                flow_img = decode_image(frame["flow"])
                cv2.imshow("cat", flow_img)

            # last_unity_time = resp["Content"]["Frame"]
            key = cv2.waitKey(1)
            print(f"real elapsed time: {time.time() - start_real_time}")
            # print(f"unity time: {last_unity_time} . elapsed: {last_unity_time - start_unity_time}")
            if key == 27:  # ESC Pressed
                break
    finally:
        print(f"Closing agent {agent.id}")
        agent.dispose()