import numpy as np
import cv2
from lite_flow_utils import run
import torch
import math


class FlowNetLiteWrapper:

    def __init__(self):
        self.prev_frame_gray_scale = None
        self.frame_gray_scale = None
        self.optical_flow = None
        self.net = run.Network().cuda().eval()

    def __call__(self, frame):
        if self.frame_gray_scale is not None:
            prev_frame_gray_scale = self.frame_gray_scale
            # if frame.ndim == 3 and frame.shape[2] == 3:
            #     frame_gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # else:
            frame_gray_scale = frame

            self.prev_frame_gray_scale = self.frame_gray_scale
            self.frame_gray_scale = frame_gray_scale

            # motion flow is estimated from frame at time t to frame at time t-1 (yes, backward...)
            # self.optical_flow = cv2.calcOpticalFlowFarneback(frame_gray_scale,
            #                                                  prev_frame_gray_scale,
            #                                                  self.optical_flow,
            #                                                  pyr_scale=0.4,
            #                                                  levels=5,  # pyramid levels
            #                                                  winsize=12,
            #                                                  iterations=10,
            #                                                  poly_n=5,
            #                                                  poly_sigma=1.1,
            #                                                  flags=0)

            # TODO remove all this garbage
            prev = np.ascontiguousarray(self.prev_frame_gray_scale[:, :, ::-1].transpose(2, 0, 1).astype(np.float32, copy=False) * (1.0 / 255.0))
            actual = np.ascontiguousarray(self.frame_gray_scale[:, :, ::-1].transpose(2, 0, 1).astype(np.float32, copy=False) * (1.0 / 255.0))

            tenFirst = torch.FloatTensor(actual)  # od invertirle?
            tenSecond = torch.FloatTensor(prev)

            intWidth = prev.shape[2]
            intHeight = prev.shape[1]

            tenPreprocessedFirst = tenFirst.cuda().view(1, 3, intHeight, intWidth)
            tenPreprocessedSecond = tenSecond.cuda().view(1, 3, intHeight, intWidth)

            intPreprocessedWidth = int(math.floor(math.ceil(intWidth / 32.0) * 32.0))
            intPreprocessedHeight = int(math.floor(math.ceil(intHeight / 32.0) * 32.0))

            tenPreprocessedFirst = torch.nn.functional.interpolate(input=tenPreprocessedFirst,
                                                                   size=(intPreprocessedHeight, intPreprocessedWidth),
                                                                   mode='bilinear', align_corners=False)
            tenPreprocessedSecond = torch.nn.functional.interpolate(input=tenPreprocessedSecond,
                                                                    size=(intPreprocessedHeight, intPreprocessedWidth),
                                                                    mode='bilinear', align_corners=False)

            tenFlow = torch.nn.functional.interpolate(input=self.net(tenPreprocessedFirst, tenPreprocessedSecond),
                                                      size=(intHeight, intWidth), mode='bilinear', align_corners=False)

            tenFlow[:, 0, :, :] *= float(intWidth) / float(intPreprocessedWidth)
            tenFlow[:, 1, :, :] *= float(intHeight) / float(intPreprocessedHeight)

            self.optical_flow = tenFlow[0, :, :, :].cpu().numpy().transpose(1, 2, 0)
        else:
            if self.frame_gray_scale is None:
                if not ((frame.ndim == 3 and frame.shape[2] == 1) or frame.ndim == 2):
                    a, b, c = frame.shape
                    self.optical_flow = np.zeros((a, b, 2), np.float32)
                    self.frame_gray_scale = frame
                else:
                    if frame.ndim == 2:
                        a, b = frame.shape
                    else:
                        a, b, c = frame.shape
                    self.optical_flow = np.zeros((a, b, 2), np.float32)
                    self.frame_gray_scale = frame

        return self.optical_flow

    @staticmethod
    def draw_flow_lines(frame, optical_flow, line_step=16, line_color=(0, 255, 0)):
        frame_with_lines = frame.copy()
        line_color = (line_color[2], line_color[1], line_color[0])

        for y in range(0, optical_flow.shape[0], line_step):
            for x in range(0, optical_flow.shape[1], line_step):
                fx, fy = optical_flow[y, x]
                cv2.line(frame_with_lines, (x, y), (int(x + fx), int(y + fy)), line_color)
                cv2.circle(frame_with_lines, (x, y), 1, line_color, -1)

        return frame_with_lines

    @staticmethod
    def draw_flow_map(frame, optical_flow):
        hsv = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
        hsv[..., 1] = 255

        mag, ang = cv2.cartToPolar(optical_flow[..., 0], optical_flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        frame_flow_map = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        return frame_flow_map