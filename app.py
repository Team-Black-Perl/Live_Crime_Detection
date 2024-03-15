from flask import Flask, render_template, request, redirect, url_for
import cv2
import time
from datetime import datetime
import numpy as np
from collections import deque
from keras.models import load_model

app = Flask(__name__)

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

        # preds = model.predict(np.expand_dims(frame, axis=0))[0]
        # Q.append(preds)

        # results = np.array(Q).mean(axis=0)
        # i = (preds > 0.50)[0]
        # prediction_history.append(i)

        # smoothed_prediction = np.mean(prediction_history) > 0.5
        # label = smoothed_prediction

        preds = model.predict(np.expand_dims(frame, axis=0))[0]
        Q.append(preds)

        results = np.array(Q).mean(axis=0)
        violence_percentage = results[0] * 100  # Assuming the index 0 corresponds to violence probability

        # Here, 'violence_percentage' represents the percentage of violence detected in the frame

        # Optionally, you can adjust the threshold to determine violence based on the percentage
        threshold_percentage = 50  # Adjust the threshold as needed

        label = violence_percentage > threshold_percentage

        text_color = (0, 255, 0)

        if label:
            text_color = (0, 0, 255)

            if not violence_detected:
                violence_detected = True
                violence_start_frame = frame_count
                violence_start_time = time.time()
        else:
            violence_detected = False

        if violence_detected and frame_count == violence_start_frame + 30:
            # Capture the current timestamp
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Send the alert with timestamp to the Telegram group
            message = f"Violence detected at {current_time}"
            with open('alert_frame.jpg', 'wb') as f:
                cv2.imwrite('alert_frame.jpg', frame * 255)

        # text = "Violence: {}".format(label)
        text = "Violence: {:.2f}%".format(violence_percentage)

        FONT = cv2.FONT_HERSHEY_SIMPLEX

        cv2.putText(output, text, (35, 50), FONT, 1.25, text_color, 3)

        # Write the frame with annotations to the output video
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
