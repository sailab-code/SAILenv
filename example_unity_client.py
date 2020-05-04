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
import cv2
import colorsys

# Import src

from old_http.agent import Agent

frames: int = 1000


def decode_image(array: np.ndarray):
    """
    Decode the given numpy array with OpenCV.

    :param array: the numpy array to decode
    :return: the decoded image that can be displayed
    """
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    return image


fromMin = 0
fromMax = 255
toMin = -1
toMax = 1


def decode_flow(array: np.ndarray):
    def remap(value):
        ret = value - fromMin
        ret = ret / (fromMax - fromMin)
        ret = ret * (toMax - toMin)
        ret = ret + toMin
        return ret

    return np.vectorize(remap)(array)


def HSVFlow(array: np.ndarray):
    def hsv(value):
        r = value[0]
        g = value[1]
        v = np.sqrt(r ** 2 + g ** 2)
        h = np.arctan2(g, r) * 57.296
        if h < 0:
            h = h + 360
        s = 1
        return np.asarray([np.floor(x * 255) for x in colorsys.hsv_to_rgb(h, s, v)], dtype=np.uint8)
    a = array.copy()
    for i, pixel in enumerate(array):
        a[i] = hsv(pixel)

    return a


if __name__ == '__main__':
    print("Generating agent...")
    agent = Agent(depth_frame_active=True,
                  flow_frame_active=False,
                  object_frame_active=False,
                  main_frame_active=True,
                  category_frame_active=True, width=256, height=192, host="127.0.0.1", port=8081)
    print("Registering agent on server...")
    agent.register()
    print(f"Agent registered with ID: {agent.id}")
    last_unity_time: float = 0.0
    # print(agent.get_resolution())
    try:
        print("Press ESC to close")
        while True:
            start_real_time = time.time()
            start_unity_time = last_unity_time
            frame = agent.get_frame()

            if frame["main"] is not None:
                main_img = frame["main"]
                cv2.imshow("PBR", main_img)

            if frame["category"] is not None:
                cat_img = np.ndarray((agent.height * agent.width, 3), dtype=np.uint8)
                for idx, sup in enumerate(frame["category"]):
                    if sup == 0:
                        cat_img[idx] = [0, 0, 255]
                    elif sup == 1:
                        cat_img[idx] = [255, 0, 0]
                    elif sup == 2:
                        cat_img[idx] = [0, 255, 0]
                    else:
                        cat_img[idx] = [0, 0, 0]

                cat_img = np.reshape(cat_img, (agent.height, agent.width, 3))

                # unity stores the image as left to right, bottom to top
                # while CV2 reads it left to right, top to bottom
                # a flip up-down solves the problem
                #cat_img = np.flipud(cat_img)
                cv2.imshow("Category", cat_img)

            #if frame["category_debug"] is not None:
            #    cat_img = decode_image(frame["category_debug"])
            #    cv2.imshow("Category Debug", cat_img)

            if frame["object"] is not None:
                obj_img = decode_image(frame["object"])
                cv2.imshow("Object ID", obj_img)

            if frame["flow"] is not None:
                flow = frame["flow"]
                cv2.imshow("Optical Flow", flow)

            if frame["depth"] is not None:
                depth = frame["depth"]
                cv2.imshow("Depth", depth)

            key = cv2.waitKey(100)
            print(f"FPS: {1/(time.time() - start_real_time)}")
            if key == 27:  # ESC Pressed
                break
    finally:
        print(f"Closing agent {agent.id}")
        #agent.delete()
