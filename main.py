import base64

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
    camera_api = CameraApi()

    while True:
        resp = camera_api.get_main_frame()
        main_img = decode_image(resp["Content"])
        cat_img = decode_image(camera_api.get_category_frame()["Content"])
        obj_img = decode_image(camera_api.get_object_frame()["Content"])

        cv2.imshow("main", main_img)
        cv2.imshow("cat", cat_img)
        cv2.imshow("obj", obj_img)

        key = cv2.waitKey(10)
        if key == 27:  # ESC Pressed
            return


if __name__ == '__main__':
    main()
