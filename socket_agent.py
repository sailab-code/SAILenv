#
# Copyright (C) 2020 Enrico Meloni, Luca Pasqualini
# University of Siena - Artificial Intelligence Laboratory - SAILab
#
#
# L2S is licensed under a MIT license.
#
# You should have received a copy of the license along with this
# work. If not, see <https://en.wikipedia.org/wiki/MIT_License>.

# Import packages

import numpy as np
import cv2
import socket
from enum import IntFlag
import gzip


# Import src

class FrameFlags(IntFlag):
    NONE = 0
    MAIN = 1
    CATEGORY = 1 << 2
    OBJECT = 1 << 3
    OPTICAL = 1 << 4
    DEPTH = 1 << 5


class SocketAgent:

    sizeof_int = 4 # sizeof_int in C#

    """
    TODO: summary ??? Maybe some more check to avoid connection errors?

    """

    def __init__(self,
                 main_frame_active: bool = True,
                 object_frame_active: bool = True,
                 category_frame_active: bool = True,
                 flow_frame_active: bool = True,
                 depth_frame_active: bool = True,
                 host: str = "localhost",
                 port: int = 8080,
                 width: int = 512,
                 height: int = 384,
                 gzip=False):
        """

        :param main_frame_active: True if the virtual world should generate the main camera view
        :param object_frame_active: True if the virtual world should generate object instance supervisions
        :param category_frame_active: True if the virtual world should generate category supervisions
        :param flow_frame_active: True if the virtual world should generate optical flow data
        :param host: address on which the unity virtual world is listening
        :param port: port on which the unity virtual world is listening
        :param width: width of the stream
        :param height: height of the stream
        :param gzip: true if the virtual world should compress the views with gzip
        """
        self.flow_frame_active: bool = flow_frame_active
        self.category_frame_active: bool = category_frame_active
        self.object_frame_active: bool = object_frame_active
        self.main_frame_active: bool = main_frame_active
        self.depth_frame_active: bool = depth_frame_active

        self.flags = 0
        self.flags |= FrameFlags.MAIN if main_frame_active else 0
        self.flags |= FrameFlags.CATEGORY if category_frame_active else 0
        self.flags |= FrameFlags.OBJECT if object_frame_active else 0
        self.flags |= FrameFlags.DEPTH if depth_frame_active else 0
        self.flags |= FrameFlags.OPTICAL if flow_frame_active else 0

        self.id: int = -1
        self.host = host
        self.port = port
        self.width = width
        self.height = height

        # creates a TCP socket over IPv4
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        self.gzip = gzip

        # numbers are sent as little endian. TODO: check if this is also true on linux or just windows.
        self.endianness = 'little'

    def register(self):
        """
        Register the agent on the Unity server and set its id.
        """
        #connect to the unity socket
        self.socket.connect((self.host, self.port))
        self.__send_resolution()
        self.__send_gzip_setting()
        self.__receive_agent_id()


    def __send_resolution(self):
        """
        convert the resolution to bytes and send it over the socket
        """
        resolution_bytes = self.width.to_bytes(4, self.endianness) + \
                           self.height.to_bytes(4, self.endianness)
        self.socket.send(resolution_bytes)

    def __send_gzip_setting(self):
        """
        send gzip option
        """
        gzip_byte = b"\x01" if self.gzip else b"\x00"
        self.socket.send(gzip_byte)

    def __receive_int(self):
        """
        Receives an integer
        :return: the received integer
        """
        data = self.receive_bytes(self.sizeof_int)
        return int.from_bytes(data, self.endianness)

    def __receive_agent_id(self):
        """
        receives agent id
        """
        self.id = self.__receive_int()

    def __receive_string(self, str_format = "utf-8"):
        """
        receives a string
        :return: the received string
        """

        string_size = self.__receive_int()
        data = self.receive_bytes(string_size)
        return data.decode(str_format)


    def delete(self):
        """
        Delete the agent on the Unity server.
        """
        self.socket.send(b'\x01')  # x01 is the code unity expects for deleting an agent

    def get_frame(self):
        """
        Get the frame from the cameras on the Unity server.

        :return: a dict of frames indexed by keys main, object, category, flow, depth. \
                 It has also a "sizes" key containing the size in byte of the received frame. (Different from the
                 actual size of the frame when gzip compression is enabled.
        """
        # initialize the frame dictionary
        frame = {"sizes": {}}

        # encodes the flags in a single byte
        flags_bytes = self.flags.to_bytes(1, self.endianness)
        # adds the FRAME request byte.
        request_bytes = b'\x00' + flags_bytes

        self.socket.send(request_bytes)

        # start reading images from socket in the following order:
        # main, category, object, optical flow, depth
        if self.main_frame_active:
            frame_bytes, received = self.receive_next_frame_view()

            # main is a png image, so it can be read with cv2
            frame["main"] = cv2.imdecode(self.__decode_image(frame_bytes), cv2.IMREAD_COLOR)
            frame["sizes"]["main"] = received
        else:
            frame["sizes"]["main"] = 0
            frame["main"] = None

        if self.category_frame_active:
            frame_bytes, received = self.receive_next_frame_view()

            cat_frame = self.__decode_category(frame_bytes)
            cat_frame = np.reshape(cat_frame, (self.height, self.width, 3))
            cat_frame = cat_frame[:, :, 0]
            frame["category"] = cat_frame.flatten()
            frame["sizes"]["category"] = received
            # frame["category_debug"] = self.__decode_image(base64_images["CategoryDebug"])
        else:
            frame["sizes"]["category"] = 0
            frame["category"] = None
            # frame["category_debug"] = None

        if self.object_frame_active:
            frame_bytes, received = self.receive_next_frame_view()
            frame["object"] = self.__decode_image(frame_bytes)
            frame["sizes"]["object"] = received
        else:
            frame["sizes"]["object"] = 0
            frame["object"] = None

        if self.flow_frame_active:
            frame_bytes, received = self.receive_next_frame_view()
            flow = self.__decode_image(frame_bytes, np.float32)
            frame["flow"] = self.__decode_flow(flow)
            frame["sizes"]["flow"] = received
        else:
            frame["sizes"]["flow"] = 0
            frame["flow"] = None

        if self.depth_frame_active:
            frame_bytes, received = self.receive_next_frame_view()
            frame["depth"] = cv2.imdecode(self.__decode_image(frame_bytes), cv2.IMREAD_COLOR)
            frame["sizes"]["depth"] = received
        else:
            frame["sizes"]["depth"] = 0
            frame["depth"] = None

        return frame

    def receive_bytes(self, n_bytes):
        """
        Receives exactly n_bytes from the socket

        :param n_bytes: Number of bytes that should be received from the socket.

        :return an array of n_bytes bytes.
        """
        received = 0
        bytes = b''
        while received < n_bytes:
            chunk = self.socket.recv(n_bytes - received)
            received += len(chunk)
            bytes += chunk
        return bytes

    def receive_next_frame_view(self):
        """
        Receives the next frame from the socket.

        :return: a byte array containing the encoded frame view.
        """
        frame_length = int.from_bytes(self.socket.recv(4), self.endianness)  # first we read the length of the frame
        frame_bytes = self.receive_bytes(frame_length)

        if self.gzip:
            try:
                return gzip.decompress(frame_bytes), frame_length
            except:
                print("Error decompressing gzip frame: is gzip enabled on the server?")
        else:
            return frame_bytes, frame_length

    def get_categories(self):
        """
        Gets the categories list from the socket
        TODO: must be implemented on Unity side.
        :return:
        """
        return []
        # content = self.sim_api.get_categories()
        # categories = {}
        # for cat in content:
        #   categories[cat["Name"]] = cat["Id"]

        # return categories

    @staticmethod
    def __decode_image(bytes, dtype=np.uint8) -> np.ndarray:
        """
        Decode an image from the given bytes representation to a numpy array.

        :param bytes: the bytes representation of an image
        :return: the numpy array representation of an image
        """
        return np.frombuffer(bytes, dtype)

    def __decode_category(self, input) -> np.ndarray:
        """
        Decode the category supervisions from the given base64 representation to a numpy array.
        :param input: the base64 representation of categories
        :return: the numpy array containing the category supervisions
        """

        cat_frame = self.__decode_image(input)
        cat_frame = np.reshape(cat_frame, (self.height, self.width, -1))
        cat_frame = np.flipud(cat_frame)
        cat_frame = np.reshape(cat_frame, (-1))
        cat_frame = np.ascontiguousarray(cat_frame)
        return cat_frame

    def __decode_flow(self, flow_frame: np.ndarray) -> np.ndarray:
        """
        Decodes the flow frame.
        :param flow_frame: frame obtained by unity. It is a float32 array.
        :return: Decoded flow frame
        """
        flow = flow_frame

        # must be flipped upside down, because Unity returns it from bottom to top.
        flow = np.reshape(flow, (self.height, self.width, -1))
        flow = np.flipud(flow)

        # restore it as contiguous array, as flipud breaks the contiguity
        flow = np.ascontiguousarray(flow)
        return flow
