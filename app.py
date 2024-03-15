from flask import Flask, render_template, request, redirect, url_for
import cv2
import time
from datetime import datetime
import numpy as np
from collections import deque
from keras.models import load_model
import os 
app = Flask(__name__)

import os

def save_annotated_video(input_video, output_video):
    print("Loading model ...")
    model = load_model('modelnew.h5')
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

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video, fourcc, 30.0, (1280, 720))  # Adjust the resolution as needed

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
        Q.append(preds)

        results = np.array(Q).mean(axis=0)
        violence_percentage = results[0] * 100  # Assuming the index 0 corresponds to violence probability

        label = violence_percentage > 50  # Adjust the threshold as needed

        if label:
            if not violence_detected:
                violence_detected = True
                violence_start_frame = frame_count
                clip_start_frame = max(0, violence_start_frame - 5 * 30)  # 5 seconds before violence start
                vs.set(cv2.CAP_PROP_POS_FRAMES, clip_start_frame)
        else:
            if violence_detected:
                clip_end_frame = frame_count
                # Capture the current timestamp
                current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

                # Save the clip
                clip_name = os.path.join(clip_dir, f'clip_{current_time}.avi')
                clip_out = cv2.VideoWriter(clip_name, fourcc, 30.0, (W, H))
                for i in range(clip_start_frame, clip_end_frame):
                    vs.set(cv2.CAP_PROP_POS_FRAMES, i)
                    ret, frame = vs.read()
                    if ret:
                        clip_out.write(frame)
                clip_out.release()
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

        output_video_file = 'annotated_video.avi'
        save_annotated_video(input_video, output_video_file)

    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
