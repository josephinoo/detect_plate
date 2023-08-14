# -*- coding: utf-8 -*-
from PIL import Image
import json
import cv2
import base64
import requests
import keras_ocr
import numpy as np
import time
import datetime
import argparse
parser = argparse.ArgumentParser(description='Process some arguments.')
parser.add_argument('--source', type=str, help='Name argument')
args = parser.parse_args()



with open('roboflow_config.json') as f:
    config = json.load(f)

    ROBOFLOW_API_KEY = config["ROBOFLOW_API_KEY"]
    ROBOFLOW_MODEL = config["ROBOFLOW_MODEL"]
    LOCAL_SERVER = config["LOCAL_SERVER"]

    FRAMERATE = config["FRAMERATE"]
    BUFFER = config["BUFFER"]

# Local Server Link
if not LOCAL_SERVER:
    upload_url = "".join([
        "https://detect.roboflow.com/" + ROBOFLOW_MODEL,
        "/2"
        "?api_key=" + ROBOFLOW_API_KEY,
      
    ])
else:
    upload_url = "".join([
        "http://127.0.0.1:9001/" + ROBOFLOW_MODEL,
        "?access_token=" + ROBOFLOW_API_KEY,
        "&name=YOUR_IMAGE.jpg"
    ])





line_x = None
line_y = None
draw_line = True

# Definir el desplazamiento hacia la izquierda (en píxeles)
left_offset = 120
# Get webcam interface via opencv-python
source = args.source if args.source else 0
video = cv2.VideoCapture(source)
pipeline = keras_ocr.pipeline.Pipeline()


def make_request_with_retries(url, data, headers, max_retries=3, retry_delay=5):
    retries = 0
    while retries < max_retries:
        try:
            resp = requests.post(url, data=data, headers=headers, stream=True)
            return resp
        except requests.exceptions.ConnectionError:
            print("Connection error. Retrying in {} seconds...".format(retry_delay))
            time.sleep(retry_delay)
            retries += 1
    print("Max retries reached. Unable to make the request.")
    return None

# Infer via the Roboflow Infer API and return the result
def infer(ret, img):
    # Get the current image from the webcam
    global line_x, line_y
    global draw_line 
    car_crossed_line = False
    retval, buffer = cv2.imencode('.jpg', img)
    height, width, _ = img.shape
    img_str = base64.b64encode(buffer)
    if draw_line:
            if line_x is None:
                line_x = width // 2 - left_offset
                line_y = 0
            cv2.line(img, (line_x, line_y), (line_x, height), (255, 255, 255, 128), 2)


    # Get predictions from Roboflow Infer API
    resp_data = None
    resp = make_request_with_retries(upload_url, img_str, {"Content-Type": "application/x-www-form-urlencoded"})
    if resp and resp.status_code == 200:
            resp_data = resp.json()
            # Process response data and perform drawing
    else:
        print("Request failed or max retries reached.")

    resp = resp_data['predictions']
    # Draw all predictions
    for prediction in resp:
        # Save License Plate as image
        x = prediction['x']
        y = prediction['y']
        w = prediction['width']
        h = prediction['height']
        
        confidence = prediction['confidence']
        
        frame = img
        if prediction['class'] == "License_Plate":
        # Check if the car is inside the line crossing ROI
                if x <= line_x and x + w >= line_x:
                            # Draw bounding box and label for the detected car
                            color = (0, 255, 0) 
                            label = "License"
                            
                            # Check if the car crossed the line
                            if not car_crossed_line:
                                car_crossed_line = True
                                draw_line = False
                                # Save the frame at the moment the car crossed the line
                                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                                img = video.read()[1]
                                save_filename = "capture/car_{}.jpg".format(timestamp)
                                save_capture = "plate/plate_{}.jpg".format(timestamp)
                                crop_frame,plate  = getLiscensePlate(img, x, y, w, h)
                                writeOnStream(x, y, w, h,
                      prediction['class'],
                      img)
                                cv2.imwrite(save_filename, frame)
                                cv2.imwrite(save_capture,crop_frame)
                                with open('results.csv', 'a') as f:
                                   f.write(timestamp + ',' + plate + ',' + save_filename + ',' + save_capture + '\n')
    car_crossed_line = False
    draw_line = True
    return img


def writeOnStream(x, y, width, height, className, frame):
    # Draw a Rectangle around detected image
    cv2.rectangle(frame, (int(x - width / 2), int(y + height / 2)), (int(x + width / 2), int(y - height / 2)),
                  (255, 0, 0), 2)

    # Draw filled box for class name
    cv2.rectangle(frame, (int(x - width / 2), int(y + height / 2)), (int(x + width / 2), int(y + height / 2) + 35),
                  (255, 0, 0), cv2.FILLED)

    # Set label font + draw Text
    font = cv2.FONT_HERSHEY_DUPLEX

    cv2.putText(frame, className, (int(x - width / 2 + 6), int(y + height / 2 + 26)), font, 0.5, (255, 255, 255), 1)


def getLiscensePlate(frame, x, y, width, height):
    # Crop license plate
    crop_frame = frame[int(y - height / 2):int(y + height / 2), int(x - width / 2):int(x + width / 2)]
    # Save license Plate
    cv2.imwrite("plate.jpg", crop_frame)
    # Pre Process Image
    preprocessImage("plate.jpg")
    # Read image for OCR
    images = [keras_ocr.tools.read("plate.jpg")]
    # Get Predictions
    prediction_groups = pipeline.recognize(images)
    # Print the predictions
    plate = []
    for predictions in prediction_groups:
        for prediction in predictions:
            plate.append(prediction[0])
            print(prediction[0])

    plate = '|'.join(plate)
    return crop_frame , plate


def preprocessImage(image):
    # Read Image
    img = cv2.imread(image)
    # Resize Image
    img = cv2.resize(img, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)
    # Change Color Format
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Kernel to filter image
    kernel = np.ones((1, 1), np.uint8)
    # Dilate + Erode image using kernel
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)
    img = cv2.addWeighted(img, 4, cv2.blur(img, (30, 30)), -4, 128)
    # Save + Return image
    cv2.imwrite('processed.jpg', img)
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return img



if __name__ == '__main__':
    # Main loop; infers sequentially until you press "q"
    frame_count = 0
    frames_to_skip = 15
    while True:
        # On "q" keypress, exit
        if (cv2.waitKey(1) == ord('q')):
            break

        # Synchronously get a prediction from the Roboflow Infer API
        ret, image = video.read()
        if not ret:
         break

        frame_count += 1
        if frame_count % frames_to_skip != 0:
            continue
        image = infer(ret, image)
        # And display the inference results
        cv2.imshow('image', image)
    # Release resources when finished
    video.release()
    cv2.destroyAllWindows()