from flask import Flask, render_template, Response
import cv2
from ultralytics import YOLO

app = Flask(__name__)

model = YOLO("best.pt")

def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Realizar la predicción con el modelo YOLO
        results = model.predict(frame, imgsz=640, conf=0.5)
        annotated_frame = results[0].plot()

        # Codificar el fotograma en formato JPEG
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame = buffer.tobytes()

        # Generar el flujo de respuesta
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

