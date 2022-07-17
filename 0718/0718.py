from flask import Flask, render_template, Response, request, url_for, redirect
import cv2
import math
import cv2
import numpy as np
import mediapipe as mp

from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates
from rembg import remove
from functions import detectBox, OutOfRangeWarning, DistanceWarning, StraightWarning, find_skinColor_crcb, crcb_oval, get_rightEarPoint, get_leftEarPoint

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_selfie_segmentation = mp.solutions.selfie_segmentation
mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection

size = np.array([(35, 45), (28, 35), (42, 47)])
sizeIndex = 0
BG_COLOR = (255, 255, 255)  # white

app = Flask(__name__)
cap = cv2.VideoCapture(0)


def gen_frames():
    while True:
        success, frame = cap.read()
        h, w = frame.shape[:2]  # hxw = 480x640
        if not success:
            print('Ignoring empty camera frame.\n')
            continue
        else:
            # Keep a copy of the un-processed frame for final picture file output
            original_frame = frame.copy()
            # Draw a green detect box
            box_upperLeft_col, box_upperLeft_row, box_h, box_w = detectBox(h, w, frame)
            # Draw canvas for warning message
            warning_message1 = np.zeros((h, w, 3), dtype='uint8')
            warning_message2 = np.zeros((h, w, 3), dtype='uint8')
            warning_message3 = np.zeros((h, w, 3), dtype='uint8')
            warning_message4 = np.zeros((h, w, 3), dtype='uint8')

            # 【Function2-1】create face detection
            with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
                # To improve performance, mark the frame as not writeable to pass by reference.
                frame.flags.writeable = False
                results = face_detection.process(frame)
                frame.flags.writeable = True

                if results.detections:
                    for detection in results.detections:
                        # mp_drawing.draw_detection(frame, detection)
                        # Following code --> based on code in draw_detection function in drawing_utils.py
                        location = detection.location_data
                        if location.HasField('relative_bounding_box'):
                            relative_bounding_box = location.relative_bounding_box
                            rect_start_point = _normalized_to_pixel_coordinates(relative_bounding_box.xmin,
                                                                                relative_bounding_box.ymin, w, h)
                            rect_end_point = _normalized_to_pixel_coordinates(
                                relative_bounding_box.xmin + relative_bounding_box.width,
                                relative_bounding_box.ymin + relative_bounding_box.height, w, h)
                            if rect_start_point and rect_end_point:
                                OutOfRangeWarning(warning_message1, rect_start_point, rect_end_point, box_upperLeft_col,
                                                  box_upperLeft_row, box_h, box_w)

                # 【Function3-1】 get edge of head image
                with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as selfie_segmentation:
                    frameTmp = original_frame
                    # Convert the BGR frame to RGB
                    frameTmp = cv2.cvtColor(frameTmp, cv2.COLOR_BGR2RGB)
                    # To improve performance, mark the frame as not writeable to pass by reference.
                    frameTmp.flags.writeable = False
                    results = selfie_segmentation.process(frameTmp)
                    frameTmp.flags.writeable = True
                    frameTmp = cv2.cvtColor(frameTmp, cv2.COLOR_RGB2BGR)

                    # Draw selfie segmentation on background
                    condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.7
                    bg_frame = np.zeros(frameTmp.shape, dtype=np.uint8)
                    bg_frame[:] = BG_COLOR
                    segmented_Tmp = np.where(condition, frameTmp, bg_frame)
                    canny = cv2.Canny(segmented_Tmp, 150, 200)

                    # 【Function3-2】 find top position of the head
                    top_position = None
                    for row in range(h):
                        for column in range(w):
                            if canny[row][column] == 255:
                                top_position = [row, column]
                                break
                        if top_position is not None:
                            break
                # 【Function3-3】 get edge of head image
                if rect_start_point and rect_end_point:
                    DistanceWarning(warning_message2, rect_start_point, rect_end_point, top_position, box_h)

            landmarks_list = []
            # 【Function4】create face mesh (for straight direction of head)
            with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True,
                                       min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
                frameTmp = original_frame
                # Convert the BGR frame to RGB
                frameTmp = cv2.cvtColor(frameTmp, cv2.COLOR_BGR2RGB)
                # To improve performance, mark the frame as not writeable to pass by reference.
                frameTmp.flags.writeable = False
                results = face_mesh.process(frameTmp)
                frameTmp.flags.writeable = True
                frameTmp = cv2.cvtColor(frameTmp, cv2.COLOR_RGB2BGR)

                if results.multi_face_landmarks:
                    face_landmarks = results.multi_face_landmarks[0]
                    for i in range(478):
                        real_x = int(face_landmarks.landmark[i].x * w)
                        real_y = int(face_landmarks.landmark[i].y * h)
                        landmarks_list.append((real_x, real_y))
            '''
            # At the same time, we get skin color image here first for Function 5 later
            cr, cb = find_skinColor_crcb(landmarks_list, original_frame)
            skinImg = crcb_oval(original_frame, cr, cb)
            '''

            if len(landmarks_list) == 478:
                upper_x = landmarks_list[10][0]
                upper_y = landmarks_list[10][1]
                lower_x = landmarks_list[152][0]
                lower_y = landmarks_list[152][1]
                if upper_x and upper_y and lower_x and lower_y:
                    StraightWarning(warning_message3, upper_x, upper_y, lower_x, lower_y)
                '''
                # 【Function5】find specific landmarks of face mesh(for ear warning)
                # Right Ear
                rightEarPoints = get_rightEarPoint(landmarks_list)
                count = 0
                for point in rightEarPoints:
                    if h > point[0] >= 0 and w > point[1] >= 0:
                        if skinImg[point[1]][point[0]].any() != 0:
                            count += 1
                if count / len(rightEarPoints) >= 0.5:
                    print("have right ear")
                else:
                    print("no right ear")
                # Left Ear
                leftEarPoints = get_leftEarPoint(landmarks_list)
                count = 0
                for point in leftEarPoints:
                    if h > point[0] >= 0 and w > point[1] >= 0:
                        if skinImg[point[1]][point[0]].any() != 0:
                            count += 1
                if count / len(leftEarPoints) >= 0.5:
                    print("have left ear")
                else:
                    print("no left ear")
                '''
            # 【Function6】create front detection
            with mp_face_mesh.FaceMesh(min_detection_confidence=0.5,
                                       min_tracking_confidence=0.5) as front_detection:
                image = original_frame.copy()
                # Flip the image horizontally for a later selfie-view display
                # Also convert the color space from BGR to RGB
                image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
                # To improve performance
                image.flags.writeable = False
                # Get the result
                results = front_detection.process(image)
                # To improve performance
                image.flags.writeable = True
                # Convert the color space from RGB to BGR
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                img_h, img_w, img_c = image.shape
                face_3d = []
                face_2d = []

                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        for idx, lm in enumerate(face_landmarks.landmark):
                            if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                                if idx == 1:
                                    nose_2d = (lm.x * img_w, lm.y * img_h)
                                    nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 8000)
                                x, y = int(lm.x * img_w), int(lm.y * img_h)
                                # Get the 2D Coordinates
                                face_2d.append([x, y])
                                # Get the 3D Coordinates
                                face_3d.append([x, y, lm.z])
                                # Convert it to the NumPy array
                        face_2d = np.array(face_2d, dtype=np.float64)
                        # Convert it to the NumPy array
                        face_3d = np.array(face_3d, dtype=np.float64)
                        # The camera matrix
                        focal_length = 1 * img_w
                        cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                               [0, focal_length, img_w / 2],
                                               [0, 0, 1]])

                        # The Distance Matrix
                        dist_matrix = np.zeros((4, 1), dtype=np.float64)
                        # Solve PnP
                        success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
                        # Get rotational matrix
                        rmat, jac = cv2.Rodrigues(rot_vec)
                        # Get angles
                        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
                        # Get the y rotation degree
                        x = angles[0] * 360
                        y = angles[1] * 360
                        # print(y)
                        # See where the user's head tilting
                        if y < -10:
                            text = "Look Right"  # Looking Left
                            # threading.Thread(target=speak, args=("Look Right",)).start()
                        elif y > 10:
                            text = "Look Left"  # Looking Right
                            # threading.Thread(target=speak, args=("Look Left",)).start()
                        elif x < -10:
                            text = "Look Up"  # Looking Down
                            # threading.Thread(target=speak, args=("Look Up",)).start()
                        elif x > 10:
                            text = "Look Down"  # Looking Up
                            # threading.Thread(target=speak, args=("Look Down",)).start()
                        else:
                            text = ""
                        '''
                        # Display the nose direction
                        nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)
                        p1 = (int(nose_2d[0]), int(nose_2d[1]))
                        p2 = (int(nose_3d_projection[0][0][0]), int(nose_3d_projection[0][0][1]))
                        cv2.line(image, p1, p2, (255, 0, 0), 2)
                        '''
                        # Add the text on the image
                        # print(text)
                        cv2.putText(warning_message4, text, (400, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 5, 4)

            final_dynamic_frame = frame

            # Displaying
            flipped_frame = cv2.flip(final_dynamic_frame, 1)
            display_frame = cv2.add(flipped_frame, warning_message1)
            display_frame = cv2.add(display_frame, warning_message2)
            display_frame = cv2.add(display_frame, warning_message3)
            display_frame = cv2.add(display_frame, warning_message4)

            ret, buffer = cv2.imencode('.jpg', display_frame)
            display_frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + display_frame + b'\r\n')


@app.route('/other', methods=['GET', 'POST'])
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
