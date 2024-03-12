import socket
import cv2
import numpy as np
from enum import Enum
import threading

TEST_IMAGE = "dance.jpg"
HOST = "127.0.0.1"  # The server's hostname or IP address
PORT = 30000  # The port used by the server
MB = 16384


class ImageStatus(Enum):
    OK = 0
    IMAGE_NOT_FOUND = 1
    IMAGE_UNAVAILABLE = 2
    UNDEFINED = 3


class CameraStatus(Enum):
    OK = 0
    LOCAL_CAMERA_NOT_INIT = 1
    LOCAL_CAMERA_NOT_OPEN = 2
    UNDEFINED = 3


class Image:
    def __init__(self, image: np.ndarray, success: ImageStatus):
        self.__image = image.copy()
        self.__success = success

    def copy(self):
        return Image(self.__image, self.__success)

    @property
    def image(self) -> np.ndarray:
        return self.__image

    @property
    def success(self) -> ImageStatus:
        return self.__success


class Camera:
    def __init__(self, image: str = None):
        self.__image = None
        self.__height = 0
        self.__width = 0

        if image is not None:
            try:
                self.__image = cv2.imread(image)
                self.__status = ImageStatus.OK
                self.__height = self.__image.shape[0]
                self.__width = self.__image.shape[1]
            except:
                self.__image = None
                self.__status = ImageStatus.IMAGE_NOT_FOUND
                self.__height = 0
                self.__width = 0

    @property
    def dimension(self) -> tuple:
        return self.__height, self.__width

    def click(self) -> Image:
        return Image(self.__image, self.__status)


class PoseCamera(Camera):
    def __init__(self, camera_port: int = 0):
        super().__init__()

        try:
            self.__cap = cv2.VideoCapture(camera_port)
        except:
            self.__camera_status = CameraStatus.LOCAL_CAMERA_NOT_INIT

        if self.__cap is None:
            self.__camera_status = CameraStatus.LOCAL_CAMERA_NOT_INIT
        elif not self.__cap.isOpened():
            self.__camera_status = CameraStatus.LOCAL_CAMERA_NOT_OPEN
        else:
            self.__camera_status = CameraStatus.OK

    def click(self) -> Image:
        frame = None
        has_frame = False
        stat = ImageStatus.IMAGE_NOT_FOUND
        self.__height = 0
        self.__width = 0

        if self.__camera_status == CameraStatus.OK:
            has_frame, frame = self.__cap.read()

            if has_frame:
                self.__height = frame.shape[0]
                self.__width = frame.shape[1]
                stat = ImageStatus.OK

        return Image(frame, stat)

    def release(self):
        self.__cap.release()


class SocketCamera(Camera):
    def __init__(self, host: str, port: int):
        super().__init__()
        self.__image = Image(np.empty((1, 1, 1)), ImageStatus.UNDEFINED)
        self.__host = host
        self.__port = port
        self.__rec_thread = threading.Thread(target=self.__receive, args=(1,), daemon=True)

    def start(self):
        self.__rec_thread.start()

    def __receive(self):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((self.__host, self.__port))
                buffer = list()
                while True:
                    data = s.recv(MB)

                    if data != 0:
                        buffer.append(data)
                    else:
                        if len(buffer) > 0:
                            image = cv2.imdecode(np.array(buffer),
                                                 cv2.IMREAD_COLOR)
                            self.__image = Image(image, ImageStatus.OK)
                            buffer = list()
        except:
            self.__image = Image(np.empty((1, 1, 1)), ImageStatus.UNDEFINED)

    def click(self) -> Image:
        return self.__image.copy()

    def release(self):
        self.__rec_thread.join()
