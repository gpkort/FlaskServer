from flask import Flask
from flask_cors import CORS
from pose import ImageData, Pose, Segment, Point
import cv2 as cv
import numpy as np
from Pose.PoseModel import ModelData, NetworkModel, PoseCamera
from Pose.Poser import PoseCalculator
import models.coco.pairs as coco

app = Flask(__name__)
CORS(app)
camera = PoseCamera(0)

pose_det = NetworkModel(coco.PROTO_PATH, coco.CAFFE_PATH)
pose_det.init_network()
pose_calculator = PoseCalculator(coco.COCO_MODEL)


# Route for seeing a data
@app.route('/pose')
def req_pose():
    ok, img = camera.click()
    pose = Pose(1)

    if ok:
        outpoints = pose_det.get_output(img)
        pose = get_pose(outpoints, img.shape[0], img.shape[1])

    response = app.response_class(
        response=pose.to_json(),
        status=200,
        mimetype='application/json'
    )
    return response


# Route for seeing a data
@app.route('/image')
def get_image_data():
    stat = 0 if camera.status else 1
    ht, wd = camera.dimension

    response = app.response_class(
        response=ImageData(stat, ht, wd).to_json(),
        status=200,
        mimetype='application/json'
    )
    return response


if __name__ == '__main__':
    app.run()

    camera.release()
    cv.destroyAllWindows()
