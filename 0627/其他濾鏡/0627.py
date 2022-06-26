import cv2
import numpy as np
import keyboard
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

cap = cv2.VideoCapture(0)

lipstick_color_flag = True


def empty(v):
    pass


# trackbars for changing lipstick color
cv2.namedWindow("change lipstick color")
cv2.resizeWindow("change lipstick color", 480, 120)
cv2.createTrackbar("Red", "change lipstick color", 0, 255, empty)
cv2.createTrackbar("Green", "change lipstick color", 0, 255, empty)
cv2.createTrackbar("Blue", "change lipstick color", 0, 255, empty)


def lipstick_color(img, lms, upper_lip, lower_lip, bgr):

    mask = np.zeros_like(img)
    lms = np.array(lms)
    points_upper_lip = lms[upper_lip, :]
    points_lower_lip = lms[lower_lip, :]
    mask = cv2.fillPoly(mask, [points_upper_lip], (255, 255, 255))
    mask = cv2.fillPoly(mask, [points_lower_lip], (255, 255, 255))

    mask = cv2.flip(mask, 1)
    img_lipstick = np.zeros_like(img)
    img_lipstick[:] = bgr
    img_lipstick = cv2.bitwise_and(mask, img_lipstick)

    img_lipstick = cv2.GaussianBlur(img_lipstick, (5, 5), 10)
    img_lipstick = cv2.addWeighted(img, 1, img_lipstick, 0.2, 0)

    return img_lipstick


while True:
    # keyboard event
    if keyboard.is_pressed('p'):
        lipstick_color_flag = not lipstick_color_flag

    ret, frame = cap.read()
    h, w = frame.shape[:2]  # hxw = 480x640


    if ret:
        landmarks_list = []
        with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True,
                                   min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
            frameTmp = frame
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

        # Displaying
        display_frame = cv2.flip(frame, 1)
        # 【Function6】change lipstick color through trackbars' RGB values
        if lipstick_color_flag:
            index_upper_lip = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 308, 415, 310, 311, 312, 13, 82, 80,
                               191, 78, 61]
            index_lower_lip = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 291, 375, 321, 405, 314, 17, 84,
                               181, 91, 146, 61, 78]
            red = cv2.getTrackbarPos("Red", "change lipstick color")
            green = cv2.getTrackbarPos("Green", "change lipstick color")
            blue = cv2.getTrackbarPos("Blue", "change lipstick color")
            color = (blue, green, red)
            display_frame = lipstick_color(display_frame, landmarks_list, index_upper_lip, index_lower_lip, color)

        cv2.imshow('camera', display_frame)

    # return value of reading the frame (ret) = False
    else:
        print('Ignoring empty camera frame.\n')
        continue

    # close the window -> esc
    if cv2.waitKey(1) == 27:
        break

cap.release()
