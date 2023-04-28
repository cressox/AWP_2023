from flask import Flask, jsonify, request
from flask_cors import CORS

import mediapipe as mp
import cv2
import numpy as np

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/')
def hello():
    return 'Hello, World!'


mp_face_detection = mp.solutions.face_detection

@app.route('/check-face', methods=['POST'])
def check_face():
    image_data = request.files['image'].read()
    nparr = np.fromstring(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Gesichtserkennung durchführen
    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    if results.detections:
        result = {'face_detected': True}
    else:
        result = {'face_detected': False}

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)