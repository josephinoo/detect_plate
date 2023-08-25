import json
import requests
import cv2
import base64

with open('roboflow_config.json') as f:
    config = json.load(f)
    ROBOFLOW_API_KEY = config["ROBOFLOW_API_KEY"]
    ROBOFLOW_MODEL = config["ROBOFLOW_MODEL"]
    LOCAL_SERVER = config["LOCAL_SERVER"]
    FRAMERATE = config["FRAMERATE"]
    BUFFER = config["BUFFER"]
    ROBOFLOW_SIZE = config["ROBOFLOW_SIZE"]
    ROBOFLOW_VERSION = config["ROBOFLOW_VERSION"]
    ROBOFLOW_MODEL2 = config["ROBOFLOW_MODEL2"]



upload_url = "".join([
        "https://detect.roboflow.com/" + ROBOFLOW_MODEL,
        "/2"+
        "?api_key=" + ROBOFLOW_API_KEY,])

def get_predictions(requests, img_str):
    resp_data = requests.post(upload_url, data=img_str, headers={
        "Content-Type": "application/x-www-form-urlencoded"
    })
    resp_data = resp_data.json()
    return resp_data['predictions']

img = cv2.imread("capture/car_2023-08-21_18:10:31.jpg")
retval, buffer = cv2.imencode('.jpg', img)


def get_direction_car(img, resp_data):
    img_str = base64.b64encode(buffer)
    resp_data = get_predictions(requests, img_str)
    if len(resp_data) > 0:
        prediction = resp_data[0]  # Assuming the first prediction is the license plate
        x = prediction['x']
        width = prediction['width']
        image_width = img.shape[1]  # Get the width of the image
        license_plate_center = x + width / 2
        half_threshold = image_width / 2
        if license_plate_center < half_threshold:
            return 0
        else:
            return 1

