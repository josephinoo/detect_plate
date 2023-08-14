# Get config variables
from PIL import Image
import json
import cv2
import base64
import requests
import keras_ocr
import numpy as np
import time
import datetime

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




# Definir las coordenadas de la línea que el carro debe cruzar (un poco a la izquierda)
line_x = None
line_y = None

# Definir el desplazamiento hacia la izquierda (en píxeles)
left_offset = 120
# Get webcam interface via opencv-python
video = cv2.VideoCapture("test3.mp4")

# keras-ocr will automatically download pretrained
# weights for the detector and recognizer.
pipeline = keras_ocr.pipeline.Pipeline()


def make_request_with_retries(url, data, headers, max_retries=3, retry_delay=5):
    retries = 0
    while retries < max_retries:
        try:
            resp = requests.post(url, data=data, headers=headers, stream=True)
            return resp
        except requests.exceptions.ConnectionError:
            print(f"Connection error. Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
            retries += 1
    print("Max retries reached. Unable to make the request.")
    return None

# Infer via the Roboflow Infer API and return the result
def infer(ret, img):
    # Get the current image from the webcam
    line_x = None
    line_y = None

    
    car_crossed_line = False
    # Resize (while maintaining the aspect ratio) to improve speed and save bandwidth
    # Encode image to base64 string
    retval, buffer = cv2.imencode('.jpg', img)
    height, width, _ = img.shape
    img_str = base64.b64encode(buffer)
    if line_x is None:
        line_x = width // 2 - left_offset
        line_y = 0
    cv2.line(img, (line_x, line_y), (line_x, height), (255, 0, 0), 2)

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
                                # Save the frame at the moment the car crossed the line
                                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                                save_filename = f"car_capture_{timestamp}.jpg"
                                cv2.imwrite(save_filename, frame)        
                                getLiscensePlate(img, x, y, w, h)
                                writeOnStream(x, y, w, h,
                      prediction['class'],
                      img)
        
      
    car_crossed_line = False
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
    for predictions in prediction_groups:
        for prediction in predictions:
            print(prediction[0])


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
    frames_to_skip = 20
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