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
import random

# Import src

from sailenv.agent import Agent
from sailenv.utilities import draw_flow_map


def decode_image(array: np.ndarray):
    """
    Decode the given numpy array with OpenCV.

    :param array: the numpy array to decode
    :return: the decoded image that can be displayed
    """
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    return image


frames: int = 1000
host = "127.0.0.1"
if __name__ == '__main__':
    print("Generating agent...")
    agent = Agent(depth_frame_active=False,
                  flow_frame_active=False,
                  object_frame_active=False,
                  main_frame_active=True,
                  category_frame_active=False, width=256, height=192, host=host, port=8085, use_gzip=False)
    print("Registering agent on server...")
    agent.register()
    print(f"Agent registered with ID: {agent.id}")
    last_unity_time: float = 0.0

    print(f"Available scenes: {agent.scenes}")

    current_scene_index = 1
    spawned_object_id = None

    scene = agent.scenes[current_scene_index]
    print(f"Changing scene to {scene}")
    agent.change_scene(scene)

    print(f"Available categories: {agent.categories}")

    print(f"Available spawnable objects: {agent.spawnable_objects_names}")

    print(f"Available lights: {agent.lights_names}")

    agent.change_main_camera_clear_flags(255, 255, 255)
    print("Changing main camera clear flags to white")

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

                cat_img = np.reshape(out, (agent.height, agent.width, 3))
                cat_img = cat_img.astype(np.uint8)

                step_get_cat = time.time() - start_get_cat
                print(f"Plot category in : {step_get_cat}")
                cv2.imshow("Category", cat_img)

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
            # print(f"FPS: {1/(time.time() - start_real_time)}")
            if key == 27:  # ESC Pressed
                break
            # Change scene (in order loop)
            if key == ord('c'):
                current_scene_index += 1
                if current_scene_index > len(agent.scenes):
                    current_scene_index = 0
                # Set the clear flags to the appropriate color
                if current_scene_index == 1:
                    agent.change_main_camera_clear_flags(255, 255, 255)
                    print("Changing main camera clear flags to white")
                else:
                    agent.change_main_camera_clear_flags(-1, -1, -1)
                    print("Changing main camera clear flags to skybox")
                scene = agent.scenes[current_scene_index]
                agent.change_scene(scene)
            # Change light intensity (decreasing in loop)
            if key == ord('l'):
                if agent.lights_names is not None and len(agent.lights_names) > 0:
                    light_name = agent.lights_names[0]
                    intensity = agent.get_light_intensity(light_name) - 0.25
                    if intensity <= 0.0:
                        agent.set_light_intensity(light_name, 1.0)
                    else:
                        agent.set_light_intensity(light_name, intensity)
            # Change ambient light color (cycling in a set of colors by random)
            if key == ord('a'):
                ambient_lights = [(255, 255, 255), (0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255)]
                random_ambient_light = random.choice(ambient_lights)
                agent.set_ambient_light_color(random_ambient_light[0], random_ambient_light[1], random_ambient_light[2])
            # Choose a random object to spawn if there is at least one available and despawn any already spawned object
            if key == ord('s'):
                if spawned_object_id is not None:
                    agent.despawn_object(spawned_object_id)
                if agent.spawnable_objects_names is not None and len(agent.spawnable_objects_names) > 0:
                    object_name = random.choice(agent.spawnable_objects_names)
                    spawned_object_id = agent.spawn_object(object_name)
    finally:
        print(f"Closing agent {agent.id}")
        # Remove the object spawned by this agent, if any
        if spawned_object_id is not None:
            agent.despawn_object(spawned_object_id)
        agent.delete()
