import cv2 as cv
import numpy as np
from pose import ImageData, Pose, Segment, Point

BODY_PARTS = {"Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
              "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
              "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
              "LEye": 15, "REar": 16, "LEar": 17, "Background": 18}

POSE_PAIRS = [["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
              ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
              ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
              ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
              ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]]


def get_pose(outpoints: np.ndarray,
             height: int,
             width: int,
             threshold: float = 0.2) -> Pose:
    pose = Pose(1, [])
    points = get_points(outpoints, height, width)
    segments = get_segment_vectors(points)

    if len(segments) > 0:
        pose.segments_from_list(segments)
        pose.status = 0

    return pose


def do_all(points: list, img: np.ndarray):
    for pair in POSE_PAIRS:
        part_from = pair[0]
        part_to = pair[1]
        id_from = BODY_PARTS.get(part_from)
        id_to = BODY_PARTS.get(part_to)

        if id_from and id_to and points[id_from] and points[id_to]:
            cv.line(img, points[id_from], points[id_to],
                    (0, 255, 0), 3)
            cv.ellipse(img,
                       points[id_from],
                       (3, 3), 0, 0, 360, (0, 0, 255),
                       cv.FILLED)
            cv.ellipse(img,
                       points[id_to],
                       (3, 3), 0, 0, 360,
                       (0, 0, 255),
                       cv.FILLED)


def get_segment_vectors(points: list) -> list:
    seg_vecs = list()

    for pair in POSE_PAIRS:
        part_from = pair[0]
        part_to = pair[1]
        id_from = BODY_PARTS.get(part_from)
        id_to = BODY_PARTS.get(part_to)

        if id_from and id_to and points[id_from] and points[id_to]:
            seg_vecs.append([points[id_from], points[id_to]])

    return seg_vecs


def draw_points_from_segment_vectors(seg_vecs: list, img: np.ndarray):
    for points in seg_vecs:
        cv.line(img, points[0], points[1],
                (0, 255, 0), 3)
        cv.ellipse(img,
                   points[0],
                   (3, 3), 0, 0, 360, (0, 0, 255),
                   cv.FILLED)
        cv.ellipse(img,
                   points[1],
                   (3, 3), 0, 0, 360,
                   (0, 0, 255),
                   cv.FILLED)


def get_points(outpoints: np.ndarray,
               height: int,
               width: int,
               threshold: float = 0.2) -> list:
    points = []

    for i in range(len(BODY_PARTS)):
        # Slice heatmap of corresponging body's part.
        heat_map = outpoints[0, i, :, :]

        _, conf, _, point = cv.minMaxLoc(heat_map)
        x = (width * point[0]) / outpoints.shape[3]
        y = (height * point[1]) / outpoints.shape[2]
        # Add a point if it's confidence is higher than threshold.
        points.append((int(x), int(y)) if conf > threshold else None)

    return points



class PoseDetection:
    def __init__(self, model_path: str) -> None:
        self.net = cv.dnn.readNetFromTensorflow(model_path)

    def get_blob_points(self,
                        img: np.ndarray,
                        scale: float = 1.0,
                        in_width: int = 368,
                        in_height: int = 368,
                        median: tuple = (127.5, 127.5, 127.5),
                        swap: bool = True,
                        crop: bool = False) -> np.ndarray:
        self.net.setInput(cv.dnn.blobFromImage(img,
                                               scale,
                                               (in_width, in_height),
                                               median,
                                               swapRB=swap,
                                               crop=crop))
        out = self.net.forward()
        # MobileNet output [1, 57, -1, -1], we only need the first 19 elements
        return out[:, :19, :, :]
