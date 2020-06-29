import numpy as np
import cv2
#from old_http.agent import Agent
from sailenv.agent import Agent
import time

fromMin = 0
fromMax = 255
toMin = -1
toMax = 1


def draw_flow_map(optical_flow):
    hsv = np.zeros((optical_flow.shape[0], optical_flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 1] = 255

    mag, ang = cv2.cartToPolar(optical_flow[..., 0], optical_flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    frame_flow_map = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return frame_flow_map

if __name__ == "__main__":
    print("Generating agent...")
    agent = Agent(depth_frame_active=False,
                  flow_frame_active=True,
                  object_frame_active=False,
                  main_frame_active=True,
                  category_frame_active=True, width=256, height=192, host="localhost", port=8085, gzip=False)
    print("Registering agent on server...")
    agent.register()
    print(f"Agent registered with ID: {agent.id}")
    last_unity_time: float = 0.0
    rebuild_flag = True
    print(f"Available scenes: {agent.scenes}")

    scene = agent.scenes[2]
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
            of = frame["flow"]
            #print(np.min(of), np.max(of))
            rebuilt_main = np.zeros(current_main.shape, current_main.dtype)
            #rebuilt_main = np.copy(current_main)
            h, w, c = current_main.shape
            # of[:,:,0] = of[:,:,0] * w
            # of[:,:,1] = of[:,:,1] * h

            flow_img = draw_flow_map(of)
            cv2.imshow("Optical Flow", flow_img)
            # of = 3 * of
            if rebuild_flag:
                for j1 in range(0, w):
                    for i1 in range(0, h):
                        if np.all(of[i1][j1] == [0,0]):
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
