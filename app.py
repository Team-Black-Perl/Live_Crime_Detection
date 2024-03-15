import threading
from flask import Flask, render_template, request, redirect, url_for
import cv2
import time
from datetime import datetime
import numpy as np
from collections import deque
from keras.models import load_model
import requests
import os

app = Flask(__name__)

r = requests.get('https://www.geojs.io/')
ip_req = requests.get('https://get.geojs.io/v1/ip.json')
ipAdd = ip_req.json()['ip']
print(ipAdd)

url = 'https://get.geojs.io/v1/ip/geo/'+ipAdd+'.json'
geo_req = requests.get(url)
geo_data = geo_req.json()
# print(geo_data)
print('longitude : '+geo_data['longitude'])
print('latitude : '+geo_data['latitude'])
print(geo_data['city'])
print(geo_data['region'])

model = load_model('model/modelnew.h5')


def save_clip(vs, clip_out, clip_start, clip_end):
    vs.set(cv2.CAP_PROP_POS_FRAMES, clip_start)
    for _ in range(clip_start, clip_end):
        ret, clip_frame = vs.read()
        if ret:
            clip_out.write(clip_frame)
    clip_out.release()


def save_annotated_video(input_video, output_video):
    Q = deque(maxlen=128)

    # Check if the input_video is an integer (webcam) or a filename
    if isinstance(input_video, int):
        vs = cv2.VideoCapture(input_video)
        webcam = True
    else:
        vs = cv2.VideoCapture(input_video)
        webcam = False

    (W, H) = (None, None)
    violence_detected = False
    violence_start_frame = None
    frame_count = 0
    clip_count = 0
    clip_dir = 'clips'
    os.makedirs(clip_dir, exist_ok=True)
    max_clip_duration = 15  # Maximum clip duration in seconds

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # Adjust the resolution as needed
    out = cv2.VideoWriter(output_video, fourcc, 30.0, (1280, 720))

    smoothing_window = 10  # Adjust the window size for smoothing
    prediction_history = deque(maxlen=smoothing_window)

    while True:
        (grabbed, frame) = vs.read()

        if not grabbed:
            break

        if W is None or H is None:
            (H, W) = frame.shape[:2]

        output = frame.copy()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (128, 128)).astype("float32")
        frame = frame.reshape(128, 128, 3) / 255

        preds = model.predict(np.expand_dims(frame, axis=0))[0]
        prediction_history.append(preds)

        results = np.array(prediction_history).mean(axis=0)
        # Assuming the index 0 corresponds to violence probability
        violence_percentage = results[0] * 100

        label = violence_percentage > 50  # Adjust the threshold as needed

        if label:
            if not violence_detected:
                violence_detected = True
                violence_start_frame = frame_count
        else:
            if violence_detected:
                # 5 seconds before violence
                clip_start_frame = max(0, frame_count - 10 * 30)
                # Calculate the clip end frame based on max_clip_duration
                clip_end_frame = frame_count + max_clip_duration * 30  # 10 seconds after violence

                # If the clip exceeds the current frame count, adjust the end frame
                if webcam:
                    clip_end_frame = frame_count + max_clip_duration * 30

                current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                clip_name = os.path.join(clip_dir, f'clip_{current_time}.avi')
                clip_out = cv2.VideoWriter(clip_name, fourcc, 30.0, (W, H))

                # Set video capture to start from clip_start_frame
                # vs.set(cv2.CAP_PROP_POS_FRAMES, clip_start_frame)

                # # Write frames to clip
                # for i in range(clip_start_frame, clip_end_frame):
                #     ret, frame = vs.read()
                #     if ret:
                #         clip_out.write(frame)

                clip_thread = threading.Thread(
                    target=save_clip, args=(vs, clip_out, clip_start_frame, clip_end_frame))
                clip_thread.start()

                clip_count += 1
                violence_detected = False

        text_color = (0, 255, 0) if not label else (0, 0, 255)
        text = "Violence: {:.2f}%".format(violence_percentage)
        FONT = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(output, text, (35, 50), FONT, 1.25, text_color, 3)

        out.write(output)
        cv2.imshow("Violence Detection", output)

        key = cv2.waitKey(1) & 0xFF
        frame_count += 1

        if key == ord("q"):
            break

    print("[INFO] Cleaning up...")
    vs.release()
    out.release()  # Release the output video writer
    cv2.destroyAllWindows()
    print(f"{clip_count} clips saved in {clip_dir}")


# Define the route to the homepage
@app.route('/')
def index():
    return render_template('index.html')

# Define a route to process the form submission


@app.route('/detect_crime', methods=['POST'])
def detect_crime():
    if 'source' in request.form:
        source = request.form['source']

        video_file = request.files['video_file']
        if video_file.filename != '':
            video_file.save('video.mp4')

        # You can replace '0' with the actual webcam source number if needed.
        input_video = 0 if source == 'webcam' else 'video.mp4'

        output_video_file = 'ouput.mp4'
        processing_thread = threading.Thread(
            target=save_annotated_video, args=(input_video, output_video_file))
        processing_thread.start()

    return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(debug=True)
