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

import time
import numpy as np
import cv2
from random import randint

# Import src

from sailenv.agent import Agent

frames: int = 1000


def decode_image(array: np.ndarray):
    """
    Decode the given numpy array with OpenCV.

    :param array: the numpy array to decode
    :return: the decoded image that can be displayed
    """
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    return image


def draw_flow_lines(frame, optical_flow, line_step=16, line_color=(0, 255, 0)):
        frame_with_lines = frame.copy()
        line_color = (line_color[2], line_color[1], line_color[0])

        for y in range(0, optical_flow.shape[0], line_step):
            for x in range(0, optical_flow.shape[1], line_step):
                fx, fy = optical_flow[y, x]
                cv2.line(frame_with_lines, (x, y), (int(x + fx), int(y + fy)), line_color)
                cv2.circle(frame_with_lines, (x, y), 1, line_color, -1)

        return frame_with_lines


def draw_flow_map(optical_flow):

        hsv = np.zeros((optical_flow.shape[0], optical_flow.shape[1], 3), dtype=np.uint8)
        hsv[..., 1] = 255

        mag, ang = cv2.cartToPolar(optical_flow[..., 0], optical_flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        frame_flow_map = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        return frame_flow_map

#host = "bronte.diism.unisi.it"
host = "127.0.0.1"
#host = "eliza.diism.unisi.it"


if __name__ == '__main__':
    print("Generating agent...")
    agent = Agent(depth_frame_active=False,
                  flow_frame_active=True,
                  object_frame_active=True,
                  main_frame_active=False,
                  category_frame_active=True, width=256, height=192, host=host, port=8085, gzip=False)
    print("Registering agent on server...")
    agent.register()
    print(f"Agent registered with ID: {agent.id}")
    last_unity_time: float = 0.0

    print(f"Available scenes: {agent.scenes}")

    scene = agent.scenes[2]
    print(f"Changing scene to {scene}")
    agent.change_scene(scene)

    print(f"Available categories: {agent.categories}")

    # print(agent.get_resolution())
    try:
        print("Press ESC to close")
        while True:
            start_real_time = time.time()
            start_unity_time = last_unity_time

            start_get = time.time()
            frame = agent.get_frame()
            step_get = time.time() - start_get

            print(f"get frame in seconds: {step_get}, fps: {1/step_get}")

            if frame["main"] is not None:
                main_img = frame["main"]
                cv2.imshow("PBR", main_img)

            if frame["category"] is not None:
                start_get_cat = time.time()
                # cat_img = np.zeros((agent.height * agent.width, 3), dtype=np.uint8)
                # Extract values and keys
                k = np.array(list(agent.cat_colors.keys()))
                v = np.array(list(agent.cat_colors.values()))

                mapping_ar = np.zeros((np.maximum(np.max(k)+1, 256), 3), dtype=v.dtype)
                mapping_ar[k] = v
                out = mapping_ar[frame["category"]]

                # for idx, sup in enumerate(frame["category"]):
                #     try:
                #         color = agent.cat_colors[sup]
                #         cat_img[idx] = color
                #     except KeyError:
                #         #print(f"key error on color get: {sup}")
                #         cat_img[idx] = [0,0,0]

                cat_img = np.reshape(out, (agent.height, agent.width, 3))
                cat_img = cat_img.astype(np.uint8)

                # unity stores the image as left to right, bottom to top
                # while CV2 reads it left to right, top to bottom
                # a flip up-down solves the problem
                #cat_img = np.flipud(cat_img)

                step_get_cat = time.time() - start_get_cat
                print(f"Plot category in : {step_get_cat}")
                cv2.imshow("Category", cat_img)

            #if frame["category_debug"] is not None:
            #    cat_img = decode_image(frame["category_debug"])
            #    cv2.imshow("Category Debug", cat_img)

            if frame["object"] is not None:
                obj_img = decode_image(frame["object"])
                cv2.imshow("Object ID", obj_img)

            if frame["flow"] is not None:
                flow = frame["flow"]
                flow_img = draw_flow_map(flow)
                cv2.imshow("Optical Flow", flow_img)

            if frame["depth"] is not None:
                depth = frame["depth"]
                cv2.imshow("Depth", depth)

            key = cv2.waitKey(1)
            #print(f"FPS: {1/(time.time() - start_real_time)}")
            if key == 27:  # ESC Pressed
                break
    finally:
        print(f"Closing agent {agent.id}")
        agent.delete()
