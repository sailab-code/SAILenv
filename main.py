import base64
import time

import numpy as np

from api.CameraApi import CameraApi
import cv2

try:
    from cv2 import cv2
except ImportError:
    pass

frames = 1000


def decode_image(b64_input):
    np_arr = np.frombuffer(base64.b64decode(b64_input), np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return img


def main():
    camera_api = CameraApi(object_frame_active=False, category_frame_active=True)

    last_unity_time = 0
    while True:
        start_real_time = time.time()
        start_unity_time = last_unity_time
        resp = camera_api.get_frame()
        images = resp["Content"]
        main_img = decode_image(images["Main"])
        cv2.imshow("main", main_img)

        if camera_api.category_frame_active:
            cat_img = decode_image(images["Category"])
            cv2.imshow("cat", cat_img)

        if camera_api.object_frame_active:
            obj_img = decode_image(images["Object"])
            cv2.imshow("obj", obj_img)

        if camera_api.flow_frame_active:
            flow_img = decode_image(images["Optical"])
            cv2.imshow("flow", flow_img)

        last_unity_time = resp["Content"]["Frame"]
        key = cv2.waitKey(10)
        print(f"real elapsed time: {time.time() - start_real_time}")
        print(f"unity time: {last_unity_time} . elapsed: {last_unity_time - start_unity_time}")
        if key == 27:  # ESC Pressed
            return


if __name__ == '__main__':
    main()
