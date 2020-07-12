import numpy as np
import cv2
# from old_http.agent import Agent
from sailenv.agent import Agent
import time
from lite_flow_utils import run
import torch
import math
from lite_flow_utils import FlowNetLiteWrapper
fromMin = 0
fromMax = 255
toMin = -1
toMax = 1




if __name__ == "__main__":
    print("Generating agent...")
    agent = Agent(depth_frame_active=False,
                  flow_frame_active=False,
                  object_frame_active=False,
                  main_frame_active=True,
                  category_frame_active=False, width=256, height=192, host="localhost", port=8085, gzip=False)
    print("Registering agent on server...")
    agent.register()
    print(f"Agent registered with ID: {agent.id}")
    last_unity_time: float = 0.0
    rebuild_flag = True
    print(f"Available scenes: {agent.scenes}")


    of_comp = FlowNetLiteWrapper()
    print("Loaded FlownetLite model...")
    scene = agent.scenes[0]
    print(f"Changing scene to {scene}")
    agent.change_scene(scene)
    try:
        print("Press ESC to close")
        frame = agent.get_frame()
        prev_main = frame["main"]
        while True:
            start_real_time = time.time()
            frame = agent.get_frame()
            current_main = frame["main"]
            of = of_comp(current_main)
            # print(np.min(of), np.max(of))
            rebuilt_main = np.zeros(current_main.shape, current_main.dtype)
            # rebuilt_main = np.copy(current_main)
            h, w, c = current_main.shape
            # of[:,:,0] = of[:,:,0] * w
            # of[:,:,1] = of[:,:,1] * h

            flow_img = of_comp.draw_flow_map(current_main, of)
            cv2.imshow("Optical Flow", flow_img)
            # of = 3 * of
            if rebuild_flag:
                for j1 in range(0, w):
                    for i1 in range(0, h):
                        if np.all(of[i1][j1] == [0, 0]):
                            continue
                        i0 = max(min(int(round(i1 + of[i1][j1][1])), h - 1), 0)
                        j0 = max(min(int(round(j1 + of[i1][j1][0])), w - 1), 0)
                        rebuilt_main[i0][j0] = current_main[i1][j1]
                        # rebuilt_main[i1][j1] = prev_main[i1][j1]

                cv2.imshow("Previous", prev_main)
                cv2.imshow("Rebuilt", rebuilt_main)
                cv2.imshow("Current", current_main)

            prev_main = current_main
            key = cv2.waitKey(1)
            print(f"OF ({of.min()}, {of.max()})")
            print(f"FPS: {1 / (time.time() - start_real_time)}")
            if key == 27:  # ESC Pressed
                break
    finally:
        print(f"Closing agent {agent.id}")
        agent.delete()
