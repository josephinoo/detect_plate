import pandas as pd
import json
import cv2
import base64
import requests
import numpy as np
from PIL import Image
import pytesseract
import easyocr
from tqdm import tqdm
file = "placas.csv"
df = pd.read_csv(file, header=None, names=['Fecha', 'Ruta'])
df = df.drop_duplicates()

# #     # Write class name and confidence

def save_plates(car, frame,plate):
      save_filename = "plates/" + plate + ".jpg"
      with open('plates.csv', 'a') as f:
            if  cv2.imwrite(save_filename, frame):
                f.write(plate +  ',' + car + "\n")


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

def infer(img,car):
        retval, buffer = cv2.imencode('.jpg', img)
        img_str = base64.b64encode(buffer)
        resp_data = requests.post(upload_url, data=img_str, headers={
                "Content-Type": "application/x-www-form-urlencoded"
            })
        
        resp_data = resp_data.json()
        resp = resp_data['predictions']
        for prediction in resp:
            # Save License Plate as image
            x = prediction['x']
            y = prediction['y']
            w = prediction['width']
            h = prediction['height']
            if prediction['class'] == LICENCE_PLATE:
                crop_frame, plate= getLiscensePlate(img, x, y, w, h)
                if crop_frame is not None and plate is not None and plate != "":
                    save_plates(car,crop_frame,plate)
               
              
       
                               
                                
def getLiscensePlate(frame, x, y, width, height):
    # Crop license plate
    crop_frame = frame[int(y - height / 2):int(y + height / 2), int(x - width / 2):int(x + width / 2)]
    # Save license Plate
    cv2.imwrite("plate.jpg", crop_frame)
    # Pre Process Image
    img_new = preprocessImage("plate.jpg")
    if img_new is None:
        return None, None
    # Read image for OCR
    cv2.imshow("plate", img_new)
    reader = easyocr.Reader(["en"], gpu=True)
    result = pytesseract.image_to_string(img_new,config='--psm 11')
   
    return crop_frame, result


def preprocessImage(image):
    img = cv2.imread(image)
    img = cv2.resize(img, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)
    img = cv2.addWeighted(img, 4, cv2.blur(img, (30, 30)), -4, 128)
    cv2.imwrite('processed.jpg', img)
    return img

for index, row in tqdm(df.iterrows()):

    car = row['Ruta']
    img = cv2.imread(row['Ruta'])
    if img is None:
        print("Image loading failed for:", row['Ruta'])
        continue  # Skip this iteration
    infer(img,car)
