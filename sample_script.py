import time
import numpy as np
import cv2
from random import randint

from sailenv.agent import Agent
from sailenv.utilities import draw_flow_map


def convert_color_space(array: np.ndarray):
    """
        Convert the given numpy array from RGB to GBR.

        :param array: the numpy array to convert
        :return: the converted image that can be displayed
        """
    image = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    return image

frames: int = 1000
host = "127.0.0.1"
if __name__ == '__main__':
    print("Generating agent...")
    agent = Agent(depth_frame_active=True,
                  flow_frame_active=True,
                  object_frame_active=True,
                  main_frame_active=True,
                  category_frame_active=True, width=256, height=192, host=host, port=8085, use_gzip=False)
    print("Registering agent on server...")
    agent.register()
    print(f"Agent registered with ID: {agent.id}")
    last_unity_time: float = 0.0

    print(f"Available scenes: {agent.scenes}")

    scene = agent.scenes[2]
    print(f"Changing scene to {scene}")
    agent.change_scene(scene)

    print(f"Available categories: {agent.categories}")
    try:
        print("Press ESC to close")
        while True:
            frame = agent.get_frame()

            # get RGB view
            if frame["main"] is not None:
                main_img = convert_color_space(frame["main"])
                cv2.imshow("PBR", main_img)

            # get instance segmentation view
            if frame["object"] is not None:
                obj_img = convert_color_space(frame["object"])
                cv2.imshow("Object ID", obj_img)

            # get class segmentation view
            if frame["category"] is not None:
                k = np.array(list(agent.cat_colors.keys()))
                v = np.array(list(agent.cat_colors.values()))

                # create a map from category to colors as tensor
                mapping_ar = np.zeros((np.maximum(np.max(k) + 1, 256), 3), dtype=v.dtype)
                mapping_ar[k] = v

                cat_img = mapping_ar[frame["category"]]
                cat_img = np.reshape(cat_img, (agent.height, agent.width, 3))
                cat_img = cat_img.astype(np.uint8)
                cv2.imshow("Category ID", cat_img)

            # getting optical flow view
            if frame["flow"] is not None:
                flow = frame["flow"]
                # utility for converting to HSV color space
                flow_img = draw_flow_map(flow)
                cv2.imshow("Optical Flow", flow_img)

            # getting depth view
            if frame["depth"] is not None:
                depth = frame["depth"]
                cv2.imshow("Depth", depth)

            key = cv2.waitKey(1)
            if key == 27:  # ESC Pressed
                break
    finally:
        print(f"Closing agent {agent.id}")
        agent.delete()