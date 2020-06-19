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

# Import src

from socket_agent import SocketAgent as Agent

frames: int = 1000


def decode_image(array: np.ndarray):
    """
    Decode the given numpy array with OpenCV.

    :param array: the numpy array to decode
    :return: the decoded image that can be displayed
    """
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    return image

#host = "bronte.diism.unisi.it"
host = "127.0.0.1"
#host = "eliza.diism.unisi.it"


if __name__ == '__main__':
    print("Generating agent...")
    agent = Agent(depth_frame_active=False,
                  flow_frame_active=False,
                  object_frame_active=False,
                  main_frame_active=True,
                  category_frame_active=False, width=256, height=192, host=host, port=8100, gzip=False)
    print("Registering agent on server...")
    agent.register()
    print(f"Agent registered with ID: {agent.id}")
    last_unity_time: float = 0.0

    print(f"Available categories: {agent.categories}")
    print(f"Available scenes: {agent.scenes}")

    scene = agent.scenes[1]
    print(f"Changing scene to {scene}")
    agent.change_scene(scene)

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

            key = cv2.waitKey(1)
            #print(f"FPS: {1/(time.time() - start_real_time)}")
            if key == 27:  # ESC Pressed
                break
    finally:
        print(f"Closing agent {agent.id}")
        #agent.delete()
