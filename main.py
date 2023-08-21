# -*- coding: utf-8 -*-
import asyncio
import json
import cv2
import base64
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

left_offset = args.line_left if args.line_left else 530
source = args.source if args.source else 0
video = cv2.VideoCapture(source)

async def save_image_with_label(timestamp, save_filename, frame):
      with open('placas.csv', 'a') as f:
            if  cv2.imwrite(save_filename, frame):
                f.write(timestamp +  ',' + save_filename + "\n")

async def get_predictions(requests, img_str):
    resp_data = await requests.post(upload_url, data=img_str, headers={
        "Content-Type": "application/x-www-form-urlencoded"
    })
    resp_data = resp_data.json()
    return resp_data['predictions']

async def infer(requests):
    global prediction_cache
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
  
    predictions = await get_predictions(requests, img_str)
    # Draw all predictions
    for prediction in predictions:
        x, y, w, h = prediction['x'], prediction['y'], prediction['width'], prediction['height']
        label = prediction['class']
        await writeOnStream(x, y, w, h, label, img)
        if label == LICENCE_PLATE:
                if x <= line_x and x + w >= line_x:
                            if not car_crossed_line:
                                car_crossed_line = True
                                draw_line = False
                                label= LICENSE_LABEL
                                timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
                                img = video.read()[1]
                                save_filename = "capture/car_{}.jpg".format(timestamp)
                                await save_image_with_label(timestamp, save_filename, img)
                                await asyncio.sleep(2)                        
    car_crossed_line = False
    draw_line = True
    return img

async def writeOnStream(x, y, width, height, className, frame):
    cv2.rectangle(frame, (int(x - width / 2), int(y + height / 2)), (int(x + width / 2), int(y - height / 2)),
                  (255, 0, 0), 2)
    cv2.rectangle(frame, (int(x - width / 2), int(y + height / 2)), (int(x + width / 2), int(y + height / 2) + 35),
                  (255, 0, 0), cv2.FILLED)
    cv2.putText(frame, className, (int(x - width / 2 + 6), int(y + height / 2 + 26)), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)


async def main():
    futures = []
    async with httpx.AsyncClient() as requests:
        while 1:
            if(cv2.waitKey(1) == ord('q')):
                break
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

asyncio.run(main())

video.release()
cv2.destroyAllWindows()
