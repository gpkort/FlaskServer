import json
import cv2 as cv
import numpy as np

class ImageData:
    def __init__(self, status: int, height: int, width: int):
        self.status = status
        self.height = height
        self.width = width

    def to_json(self, pretty_print: bool = False) -> str:
        if pretty_print:
            return json.dumps(self,
                              default=lambda o: o.__dict__,
                              sort_keys=True,
                              indent=4)
        return json.dumps(self, default=lambda o: o.__dict__)


class Point:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    @staticmethod
    def from_tuple(xy: tuple):
        return Point(x=xy[0], y=xy[1])

    def to_json(self, pretty_print: bool = False) -> str:
        return json.dumps(self,
                          default=lambda o: o.__dict__,
                          sort_keys=pretty_print,
                          indent=4 if pretty_print else 0)


class Segment:
    def __init__(self, start: Point, end: Point):
        self.start = start
        self.end = end

    @staticmethod
    def from_list(pt: list):
        return Segment(Point.from_tuple(pt[0]),
                       Point.from_tuple(pt[1]))

    def to_json(self, pretty_print: bool = False) -> str:
        return json.dumps(self,
                          default=lambda o: o.__dict__,
                          sort_keys=pretty_print,
                          indent=4 if pretty_print else 0)


class Pose:
    def __init__(self, status: int, segments=None):
        self.status = status
        self.segments = list() if segments is None else segments

    def add_segment(self, vec: Segment):
        self.segments.append(vec)

    def segments_from_list(self, vecs: list):
        self.segments = [Segment.from_list(v) for v in vecs]

    def to_json(self, pretty_print: bool = False) -> str:
        return json.dumps(self,
                          default=lambda o: o.__dict__,
                          sort_keys=pretty_print,
                          indent=4 if pretty_print else 0)


