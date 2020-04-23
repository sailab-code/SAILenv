import numpy as np
import cv2
from agent import Agent
import time


if __name__ == "__main__":
    print("Generating agent...")
    agent = Agent(flow_frame_active=True, object_frame_active=False, main_frame_active=True,
                  category_frame_active=False, width=200, height=150, host="127.0.0.1", port=8081)
    print("Registering agent on server...")
    agent.register()
    print(f"Agent registered with ID: {agent.id}")
    try:
        print("Press ESC to close")
        frame = agent.get_frame()
        prev_main = frame["main"]
        while True:
            start_real_time = time.time()
            frame = agent.get_frame()
            current_main = frame["main"]
            of = frame["flow"]
            rebuilt_main = np.zeros(current_main.shape, current_main.dtype)
            #rebuilt_main = np.copy(current_main)
            h, w, c = current_main.shape
            of[:,:,0] = of[:,:,0] * w
            of[:,:,1] = of[:,:,1] * h

            # of = 3 * of

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
            key = cv2.waitKey(100)
            print(f"OF ({of.min()}, {of.max()})")
            print(f"FPS: {1 / (time.time() - start_real_time)}")
            if key == 27:  # ESC Pressed
                break
    finally:
        print(f"Closing agent {agent.id}")
        agent.delete()
