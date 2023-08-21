# -*- coding: utf-8 -*-
import asyncio
from PIL import Image
import json
import cv2
import base64
import keras_ocr
import numpy as np
import httpx
import datetime
import argparse
parser = argparse.ArgumentParser(description='Process some arguments.')
parser.add_argument('--source', type=str, help='Name argument')
parser.add_argument('--line_left', type=int, help='Name argument')
args = parser.parse_args()



with open('roboflow_config.json') as f:
    config = json.load(f)

    ROBOFLOW_API_KEY = config["ROBOFLOW_API_KEY"]
    ROBOFLOW_MODEL = config["ROBOFLOW_MODEL"]
    LOCAL_SERVER = config["LOCAL_SERVER"]
    FRAMERATE = config["FRAMERATE"]
    BUFFER = config["BUFFER"]
    ROBOFLOW_SIZE = config["ROBOFLOW_SIZE"]

LICENCE_PLATE = "License_Plate"
LICENSE_LABEL = "License"
# Local Server Link
upload_url = "".join([
        "https://detect.roboflow.com/" + ROBOFLOW_MODEL,
        "/2"
        "?api_key=" + ROBOFLOW_API_KEY,])





line_x = None
line_y = None
draw_line = True

# Definir el desplazamiento hacia la izquierda (en píxeles)
left_offset = args.line_left if args.line_left else 530
# Get webcam interface via opencv-python
source = args.source if args.source else 0
video = cv2.VideoCapture(source)
pipeline = keras_ocr.pipeline.Pipeline()



# Infer via the Roboflow Infer API and return the result

async def save_image_with_label(timestamp, save_filename, frame):
      with open('placas.csv', 'a') as f:
            if  cv2.imwrite(save_filename, frame):
                f.write(timestamp +  ',' + save_filename +  '\n')
                   
async def infer(requests):
    global line_x, line_y
    global draw_line 
    car_crossed_line = False
    ret, img = video.read()
    height, width, channels = img.shape
    retval, buffer = cv2.imencode('.jpg', img)
    img_str = base64.b64encode(buffer)
    if draw_line:
            if line_x is None:
                line_x = left_offset
                line_y = 0
            cv2.line(img, (line_x, line_y), (line_x, height), (255, 255, 255, 128), 2)
        # Get prediction from Roboflow Infer API
    resp_data = await requests.post(upload_url, data=img_str, headers={
            "Content-Type": "application/x-www-form-urlencoded"
        })
    resp_data = resp_data.json()
    resp = resp_data['predictions']
    # Draw all predictions
    for prediction in resp:
        # Save License Plate as image
        x = prediction['x']
        y = prediction['y']
        w = prediction['width']
        h = prediction['height']
        frame = img
        if prediction['class'] == LICENCE_PLATE:
        # Check if the car is inside the line crossing ROI
                if x <= line_x and x + w >= line_x:
                            if not car_crossed_line:
                                car_crossed_line = True
                                draw_line = False
                                timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
                                img = video.read()[1]
                                save_filename = "capture/car_{}.jpg".format(timestamp)
                                await save_image_with_label(timestamp, save_filename, img)
                               
    car_crossed_line = False
    draw_line = True
    return img


# def writeOnStream(x, y, width, height, className, frame):
#     # Draw a Rectangle around detected image
#     cv2.rectangle(frame, (int(x - width / 2), int(y + height / 2)), (int(x + width / 2), int(y - height / 2)),
#                   (255, 0, 0), 2)

#     # Draw filled box for class name
#     cv2.rectangle(frame, (int(x - width / 2), int(y + height / 2)), (int(x + width / 2), int(y + height / 2) + 35),
#                   (255, 0, 0), cv2.FILLED)

#     # Set label font + draw Text
#     font = cv2.FONT_HERSHEY_DUPLEX

#     cv2.putText(frame, className, (int(x - width / 2 + 6), int(y + height / 2 + 26)), font, 0.5, (255, 255, 255), 1)


# def getLiscensePlate(frame, x, y, width, height):
#     # Crop license plate
#     crop_frame = frame[int(y - height / 2):int(y + height / 2), int(x - width / 2):int(x + width / 2)]
#     # Save license Plate
#     cv2.imwrite("plate.jpg", crop_frame)
#     # Pre Process Image
#     preprocessImage("plate.jpg")
#     # Read image for OCR
#     images = [keras_ocr.tools.read("plate.jpg")]
#     # Get Predictions
#     prediction_groups = pipeline.recognize(images)
#     # Print the predictions
#     plate = []
#     for predictions in prediction_groups:
#         for prediction in predictions:
#             plate.append(prediction[0])
#     plate = '|'.join(plate)
#     return crop_frame , plate


def preprocessImage(image):
    img = cv2.imread(image)
    img = cv2.resize(img, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)
    img = cv2.addWeighted(img, 4, cv2.blur(img, (30, 30)), -4, 128)
    cv2.imwrite('processed.jpg', img)
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return img



async def main():
    # Main loop; infers sequentially until you press "q"
    futures = []
    async with httpx.AsyncClient() as requests:
        while 1:
            # On "q" keypress, exit
            if(cv2.waitKey(1) == ord('q')):
                break

        # Synchronously get a prediction from the Roboflow Infer API
            task = asyncio.create_task(infer(requests))
            futures.append(task)
            if len(futures) >= BUFFER * FRAMERATE:
                image_task = futures.pop(0)
                try:
                    image = await image_task
                    if image.shape[0] > 0 and image.shape[1] > 0:
                        cv2.imshow('image', image)
                except Exception as e:
                    print("Error:", e)
    # Release resources when finished

# Run our main loop
asyncio.run(main())

# Release resources when finished
video.release()
cv2.destroyAllWindows()
