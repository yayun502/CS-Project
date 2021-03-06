from flask import Flask, render_template, Response, request, url_for, redirect
import cv2

app = Flask(__name__)
cap = cv2.VideoCapture(0)


def gen_frames():
    while True:
        success, frame = cap.read()
        if not success:
            print('Ignoring empty camera frame.\n')
            continue
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/other')
def other():
    return render_template('other.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
