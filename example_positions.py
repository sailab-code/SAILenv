from time import sleep
import random

from socket_agent import SocketAgent as Agent
import cv2

host = "127.0.0.1"
if __name__ == '__main__':
    print("Generating agent...")
    agent = Agent(depth_frame_active=False,
                  flow_frame_active=False,
                  object_frame_active=False,
                  main_frame_active=True,
                  category_frame_active=False, width=256, height=192, host=host, port=8085, gzip=False)
    print("Registering agent on server...")
    agent.register()
    print(f"Agent registered with ID: {agent.id}")
    last_unity_time: float = 0.0

    print(f"Available scenes: {agent.scenes}")

    scene = agent.scenes[1]
    print(f"Changing scene to {scene}")
    agent.change_scene(scene)
    agent.get_categories()
    print(f"Available categories: {agent.categories}")

    try:
        print("Press ESC to close")
        frame_n = 1
        while True:
            print(f"Frame: {frame_n}")
            if frame_n % 3 == 0:
                print("Random change")
                x = random.uniform(-1., 1.)
                y = random.uniform(0.5, 1.5)
                z = random.uniform(-1., 1.)

                rx = random.uniform(0., 360.)
                ry = random.uniform(0., 360.)
                rz = random.uniform(0., 360.)

                agent.set_position((x, y, z))
                agent.set_rotation((rx, ry, rz))

            print("Position: " + str(agent.get_position()))
            print("Rotation: " + str(agent.get_rotation()))
            print("")
            frame = agent.get_frame()

            if frame["main"] is not None:
                main_img = frame["main"]
                cv2.imshow("PBR", main_img)

            key = cv2.waitKey(1000)
            #print(f"FPS: {1/(time.time() - start_real_time)}")
            if key == 27:  # ESC Pressed
                break
            frame_n += 1

    finally:
        print(f"Closing agent {agent.id}")
        agent.delete()
