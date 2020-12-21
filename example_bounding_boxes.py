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

import cv2

# Import src

from sailenv.agent import Agent


def get_bboxes(cat_frame, obj_frame):
    boundaries = {}

    for y in range(0, obj_frame.shape[0]):
        for x in range(0, obj_frame.shape[1]):
            obj_id = tuple(obj_frame[y, x])
            if obj_id not in boundaries:
                boundaries[obj_id] = {
                    "top": y,
                    "bottom": y,
                    "left": x,
                    "right": x,
                    "category": cat_frame[y, x]
                }
            else:
                boundary = boundaries[obj_id]
                boundary["top"] = max(boundary["top"], y)
                boundary["bottom"] = min(boundary["bottom"], y)
                boundary["left"] = min(boundary["left"], x)
                boundary["right"] = max(boundary["right"], x)

    return boundaries.values()


host = "127.0.0.1"
if __name__ == '__main__':
    print("Generating agent...")
    agent = Agent(depth_frame_active=False,
                  flow_frame_active=False,
                  object_frame_active=True,
                  main_frame_active=True,
                  category_frame_active=True, width=256*2, height=192*2, host=host, port=8085, use_gzip=False)
    print("Registering agent on server...")
    agent.register()
    print(f"Agent registered with ID: {agent.id}")

    scene = agent.scenes[2]
    print(f"Changing scene to {scene}")
    agent.change_scene(scene)

    print(f"Available categories: {agent.categories}")

    try:
        print("Press ESC to close")
        while True:
            frame = agent.get_frame()

            bboxes = []
            if frame["object"] is not None and frame["category"] is not None:
                obj_img = cv2.imdecode(frame["object"], cv2.IMREAD_COLOR)
                bboxes = get_bboxes(frame["category"], obj_img)

            if frame["main"] is not None:
                main_img = frame["main"]

                for bbox in bboxes:
                    lb = (bbox["left"], bbox["bottom"])
                    rt = (bbox["right"], bbox["top"])
                    category = bbox["category"]
                    if category < 100:
                        color = agent.cat_colors[category]
                        cat_name = agent.categories[category]
                        cv2.rectangle(main_img, lb, rt, color, 1)
                        cv2.putText(main_img, cat_name, lb, 0, 0.3, color, thickness=1)

                cv2.imshow("PBR", main_img)
                key = cv2.waitKey(1)
                if key == 27:  # ESC Pressed
                    break

    finally:
        print(f"Closing agent {agent.id}")
        agent.delete()
