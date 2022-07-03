import math
import cv2
import numpy as np
import keyboard
import mediapipe as mp
import pyttsx3
import threading

from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates
from rembg import remove

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_selfie_segmentation = mp.solutions.selfie_segmentation
mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection

cap = cv2.VideoCapture(0)
size = np.array([(35, 45), (28, 35), (42, 47)])
sizeIndex = 0
light_enhance_flag = True
beauty_face_flag = False
segmentation_flag = True
BG_COLOR = (255, 255, 255)  # white
Alpha = 1.8
Beta = 10
s = pyttsx3.init()
s.setProperty('rate', 200)


def empty(v):
    pass


# trackbars for light-adjustment parameters alpha, beta
cv2.namedWindow("Light Adjustment")
cv2.resizeWindow("Light Adjustment", 480, 120)
cv2.createTrackbar("Alpha", "Light Adjustment", 1, 500, empty)
cv2.createTrackbar("Beta", "Light Adjustment", 0, 200, empty)


def speak(sentence):
    s.say(sentence)
    s.runAndWait()


def detectBox(height, width, image):
    times = min(math.floor(h / size[sizeIndex][1]), math.floor(w / size[sizeIndex][0]))
    # print('times = ', times)
    boxWidth = times * size[sizeIndex][0]
    boxHeight = times * size[sizeIndex][1]
    col_upper_left = (w - boxWidth) / 2
    row_upper_left = (h - boxHeight) / 2
    col_bottom_right = (w + boxWidth) / 2
    row_bottom_right = (h + boxHeight) / 2
    # 720*1280
    cv2.rectangle(image, (round(col_upper_left), round(row_upper_left)),
                  (round(col_bottom_right), round(row_bottom_right)), (0, 255, 0), 2)
    return round(col_upper_left), round(row_upper_left), height, boxWidth


def cutEdge(upperLeft, height, width, image):
    cut = image[:height, upperLeft:upperLeft + math.ceil(width)]
    return cut


def OutOfRangeWarning(image, rect_start_point, rect_end_point, box_upperLeft_col, box_upperLeft_row, box_h, box_w):
    warning_message_pos = (40, 60)
    txt = ""
    warm = False
    if rect_end_point[0] > (box_upperLeft_col + box_w):
        warm = True
        txt = 'PLEASE MOVE RIGHT'
        # cv2.putText(image, 'PLEASE MOVE RIGHT', warning_message_pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 5, 4)
    elif rect_end_point[1] > (box_upperLeft_row + box_h):
        warm = True
        txt = 'PLEASE MOVE UP'
        # cv2.putText(image, 'PLEASE MOVE UP', warning_message_pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 5, 4)
    elif rect_start_point[0] < box_upperLeft_col:
        warm = True
        txt = 'PLEASE MOVE LEFT'
        # cv2.putText(image, 'PLEASE MOVE LEFT', warning_message_pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 5, 4)
    elif rect_start_point[1] < box_upperLeft_row:
        warm = True
        txt = 'PLEASE MOVE DOWN'
        # cv2.putText(image, 'PLEASE MOVE DOWN', warning_message_pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 5, 4)

    if warm:
        threading.Thread(target=speak, args=(txt,)).start()
        cv2.putText(image, txt, warning_message_pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 5, 4)

def DistanceWarning(image, rect_start_point, rect_end_point, head_top_position, picture_h):
    # Below Directions are described in non-flipped way
    upperLeft_x = rect_start_point[0]
    upperLeft_y = head_top_position[0]
    bottomRight_x = rect_end_point[0]
    bottomRight_y = rect_end_point[1]
    # cv2.rectangle(frame, (upperLeft_x, upperLeft_y), (bottomRight_x, bottomRight_y), color=(255, 255, 255), thickness=2)

    # Too close/far warning
    message_pos = (300, 440)
    if sizeIndex == 0:
        portion = (bottomRight_y - upperLeft_y) / picture_h
        if portion > (3.6 / 4.2):
            cv2.putText(image, 'PLEASE STAY AWAY', message_pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 5, 4)
        elif portion < (3.2 / 4.2):
            cv2.putText(image, 'PLEASE GET CLOSER', message_pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 5, 4)


def StraightWarning(image, upper_x, upper_y, lower_x, lower_y):
    slope_reciprocal = abs(lower_x - upper_x) / (lower_y - upper_y)
    message_pos = (40, 440)
    if slope_reciprocal > 0.05:
        cv2.putText(image, 'Stay straight!', message_pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 5, 4)
        threading.Thread(target=speak, args=("Stay straight",)).start()


def beauty_face(img):
    v1 = 3
    v2 = 1
    dx = v1 * 5  # 双边滤波参数之一
    fc = v1 * 12.5  # 双边滤波参数之一
    p = 0.1
    temp1 = cv2.bilateralFilter(img, dx, fc, fc)
    temp2 = cv2.subtract(temp1, img);
    temp2 = cv2.add(temp2, (10, 10, 10, 128))
    temp3 = cv2.GaussianBlur(temp2, (2 * v2 - 1, 2 * v2 - 1), 0)
    temp4 = cv2.add(img, temp3)
    dst = cv2.addWeighted(img, p, temp4, 1 - p, 0.0)
    dst = cv2.add(dst, (10, 10, 10, 255))
    return dst


def light_adjust(img, alpha, beta):
    arr = np.array(img)

    for i in range(h):
        for j in range(w):
            for k in range(3):
                temp = arr[i][j][k] * alpha + beta
                if temp > 255:
                    arr[i][j][k] = 2 * 255 - temp
                else:
                    arr[i][j][k] = temp
    return arr


def crcb_oval(image, average_skinColor_cr, average_skinColor_cb):
    """
    YCrCb顏色空間的 Cr Cb分量
    優點:以該frame計算臉頰膚色的平均比較作為中心，建立橢圓的顏色區域劃分
    缺點:計算慢(主要源於橢圓範圍內判斷)
    """
    img = image
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    (y, cr, cb) = cv2.split(ycrcb)

    skinCrCbHist = np.zeros((256, 256), dtype=np.uint8)
    cv2.ellipse(skinCrCbHist, (average_skinColor_cb, average_skinColor_cr), (20, 12), 43, 0, 360, (255, 255, 255), -1)
    skin2 = np.zeros(cr.shape, dtype=np.uint8)
    (x, y) = cr.shape
    for i in range(0, x):
        for j in range(0, y):
            CR = ycrcb[i, j, 1]
            CB = ycrcb[i, j, 2]
            if skinCrCbHist[CR, CB] > 0:
                skin2[i, j] = 255
    dst2 = cv2.bitwise_and(img, img, mask=skin2)

    return dst2

    # '''
    # YCrCb顏色空間的Cr分量 + otsu分割法:自動計算最佳threshold值
    # 優點:計算快
    # 缺點:臉部光線過亮/過暗影響分割結果
    # '''
    # ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    # (y, cr, cb) = cv2.split(ycrcb)
    # cr1 = cv2.GaussianBlur(cr, (5, 5), 0)
    # _, skin = cv2.threshold(cr1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # dst = cv2.bitwise_and(img, img, mask=skin)
    # cv2.imshow("image CR", cr1)
    # # cv2.imshow("seperate", dst)
    #
    # return dst


def compute(img, min_percentile, max_percentile):
    """計算分位點，目的是去掉圖1的直方圖兩頭的異常情況"""
    max_percentile_pixel = np.percentile(img, max_percentile)  # param 2為百分比分位數的百分比
    min_percentile_pixel = np.percentile(img, min_percentile)

    return max_percentile_pixel, min_percentile_pixel


def aug(src):
    """圖像亮度增強"""
    if get_lightness(src) > 130:
        print("圖片亮度足夠，不做增強")
    else:
        print("自動調整圖片亮度")
    # 先計算分位點，去掉像素值中少數異常值，這個分位點可以自己配置。
    # 比如1中直方圖的紅色在0到255上都有值，但是實際上像素值主要在0到20內。
    max_percentile_pixel, min_percentile_pixel = compute(src, 1, 99)

    # 去掉分位值區間之外的值
    src[src >= max_percentile_pixel] = max_percentile_pixel
    src[src <= min_percentile_pixel] = min_percentile_pixel

    # 將分位值區間拉伸到0到255，這裏取了255*0.1與255*0.9是因爲可能會出現像素值溢出的情況，所以最好不要設置爲0到255。
    out = np.zeros(src.shape, src.dtype)
    cv2.normalize(src, out, 255 * 0.1, 255 * 0.9, cv2.NORM_MINMAX)
    return out


def get_lightness(src):
    # 計算亮度
    hsv_image = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    lightness = hsv_image[:, :, 2].mean()

    return lightness


def get_rightEarPoint():
    rightEar_point = []
    earRight1_real_x = landmarks_list[127][0] - 8
    earRight1_real_y = landmarks_list[127][1] - 8
    earRight2_real_x = landmarks_list[234][0] - 6
    earRight2_real_y = landmarks_list[234][1] - 6
    earRight3_real_x = landmarks_list[93][0] - 6
    earRight3_real_y = landmarks_list[93][1] - 6
    earRight4_real_x = landmarks_list[132][0] - 6
    earRight4_real_y = landmarks_list[132][1] - 6
    earRight5_real_x = landmarks_list[58][0] - 8
    earRight5_real_y = landmarks_list[58][1] - 8

    earRight_tx1 = round((earRight1_real_x + earRight2_real_x * 5) / 6)
    earRight_ty1 = round((earRight1_real_y + earRight2_real_y * 5) / 6)
    earRight_tx2 = round((earRight2_real_x + earRight3_real_x) / 2)
    earRight_ty2 = round((earRight2_real_y + earRight3_real_y) / 2)
    earRight_tx3 = round((earRight3_real_x + earRight4_real_x) / 2)
    earRight_ty3 = round((earRight3_real_y + earRight4_real_y) / 2)
    earRight_tx4 = round((earRight4_real_x * 5 + earRight5_real_x) / 6)
    earRight_ty4 = round((earRight4_real_y * 5 + earRight5_real_y) / 6)
    earRight_tx1_2 = round((earRight_tx1 + earRight_tx2) / 2)
    earRight_ty1_2 = round((earRight_ty1 + earRight_ty2) / 2)
    earRight_tx2_3 = round((earRight_tx2 + earRight_tx3) / 2)
    earRight_ty2_3 = round((earRight_ty2 + earRight_ty3) / 2)
    earRight_tx3_4 = round((earRight_tx3 + earRight_tx4) / 2)
    earRight_ty3_4 = round((earRight_ty3 + earRight_ty4) / 2)

    rightEar_point.append((earRight_tx1, earRight_ty1))
    rightEar_point.append((earRight_tx2, earRight_ty2))
    rightEar_point.append((earRight_tx3, earRight_ty3))
    rightEar_point.append((earRight_tx4, earRight_ty4))
    rightEar_point.append((earRight_tx1_2, earRight_ty1_2))
    rightEar_point.append((earRight_tx2_3, earRight_ty2_3))
    rightEar_point.append((earRight_tx3_4, earRight_ty3_4))
    rightEar_point.append((earRight_tx1 - 2, earRight_ty1 + 5))
    rightEar_point.append((earRight_tx2 - 2, earRight_ty2 + 5))
    rightEar_point.append((earRight_tx3 - 2, earRight_ty3 - 5))
    rightEar_point.append((earRight_tx1_2 - 2, earRight_ty1_2 + 5))
    rightEar_point.append((earRight_tx2_3 - 1, earRight_ty2_3 + 5))

    return rightEar_point


def get_leftEarPoint():
    leftEar_point = []
    earLeft1_real_x = landmarks_list[356][0] + 8
    earLeft1_real_y = landmarks_list[356][1] + 8
    earLeft2_real_x = landmarks_list[454][0] + 6
    earLeft2_real_y = landmarks_list[454][1] + 6
    earLeft3_real_x = landmarks_list[323][0] + 6
    earLeft3_real_y = landmarks_list[323][1] + 6
    earLeft4_real_x = landmarks_list[361][0] + 6
    earLeft4_real_y = landmarks_list[361][1] + 6

    earLeft_tx1 = round((earLeft1_real_x + earLeft2_real_x) / 2)
    earLeft_ty1 = round((earLeft1_real_y + earLeft2_real_y) / 2)
    earLeft_tx2 = round((earLeft2_real_x + earLeft3_real_x) / 2)
    earLeft_ty2 = round((earLeft2_real_y + earLeft3_real_y) / 2)
    earLeft_tx3 = round((earLeft3_real_x + earLeft4_real_x) / 2)
    earLeft_ty3 = round((earLeft3_real_y + earLeft4_real_y) / 2)
    earLeft_tx1_2 = round((earLeft_tx1 + earLeft_tx2) / 2)
    earLeft_ty1_2 = round((earLeft_ty1 + earLeft_ty2) / 2)
    earLeft_tx2_3 = round((earLeft_tx2 + earLeft_tx3) / 2)
    earLeft_ty2_3 = round((earLeft_ty2 + earLeft_ty3) / 2)

    leftEar_point.append((earLeft_tx1, earLeft_ty1))
    leftEar_point.append((earLeft_tx2, earLeft_ty2))
    leftEar_point.append((earLeft_tx3, earLeft_ty3))
    leftEar_point.append((earLeft_tx1_2, earLeft_ty1_2))
    leftEar_point.append((earLeft_tx2_3, earLeft_ty2_3))
    leftEar_point.append((earLeft_tx1 + 2, earLeft_ty1 - 5))
    leftEar_point.append((earLeft_tx2 + 2, earLeft_ty2 - 5))
    leftEar_point.append((earLeft_tx3 + 2, earLeft_ty3 - 5))
    leftEar_point.append((earLeft_tx1_2 + 2, earLeft_ty1_2 - 5))
    leftEar_point.append((earLeft_tx2_3 + 1, earLeft_ty2_3 - 5))

    return leftEar_point


def find_skinColor_crcb():
    upper_left_x = landmarks_list[340][0]
    upper_left_y = landmarks_list[340][1]
    lower_left_x = landmarks_list[322][0]
    lower_left_y = landmarks_list[322][1]
    upper_right_x = landmarks_list[120][0]
    upper_right_y = landmarks_list[120][1]
    lower_right_x = landmarks_list[215][0]
    lower_right_y = landmarks_list[215][1]
    Count = 0
    total = [0, 0, 0]
    for x in range(lower_left_x, upper_left_x):
        for y in range(lower_left_y, upper_left_y):
            total = np.add(total, original_frame[x][y])
            Count = Count + 1
    for x in range(lower_right_x, upper_right_x):
        for y in range(upper_right_y, lower_right_y):
            total = np.add(total, original_frame[x][y])
            Count = Count + 1
    average_skinColor_rgb = [round(total[0] / Count), round(total[1] / Count), round(total[2] / Count)]

    skinColor_rgb = np.zeros((1, 1, 3), dtype='uint8')
    for x in range(skinColor_rgb.shape[0]):
        for y in range(skinColor_rgb.shape[1]):
            skinColor_rgb[x][y] = np.array(average_skinColor_rgb)
    skinColor_ycrcb = cv2.cvtColor(skinColor_rgb, cv2.COLOR_BGR2YCR_CB)
    (skinColor_y, skinColor_cr, skinColor_cb) = cv2.split(skinColor_ycrcb)

    return skinColor_cr[0][0], skinColor_cb[0][0]


def add_white_bg(img):
    canvas = np.ones((h, w, 3), dtype="uint8")
    canvas *= 255
    person = remove(img)
    for i in range(w):
        canvas[:, i, 0] = canvas[:, i, 0] * (1 - person[:, i, 3] / 255) + person[:, i, 0] * (person[:, i, 3] / 255)
        canvas[:, i, 1] = canvas[:, i, 1] * (1 - person[:, i, 3] / 255) + person[:, i, 1] * (person[:, i, 3] / 255)
        canvas[:, i, 2] = canvas[:, i, 2] * (1 - person[:, i, 3] / 255) + person[:, i, 2] * (person[:, i, 3] / 255)
    return canvas


while True:
    # keyboard event
    if keyboard.is_pressed('1'):
        sizeIndex = 0
    elif keyboard.is_pressed('2'):
        sizeIndex = 1
    elif keyboard.is_pressed('3'):
        sizeIndex = 2
    if keyboard.is_pressed('l'):
        if light_enhance_flag:
            print('stop light enhancement function')
        else:
            print('using light enhancement function')
        light_enhance_flag = not light_enhance_flag
    if keyboard.is_pressed('b') | keyboard.is_pressed('B'):
        if beauty_face_flag:
            print('stop beauty face function')
        else:
            print('using beauty face function')
        beauty_face_flag = not beauty_face_flag
    if keyboard.is_pressed('s'):
        if segmentation_flag:
            print('stop segmentation function')
        else:
            print('using segmentation function')
        segmentation_flag = not segmentation_flag

    ret, frame = cap.read()
    h, w = frame.shape[:2]  # hxw = 480x640

    # drawing spec for face mesh
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    if ret:
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
                        rect_start_point = _normalized_to_pixel_coordinates(
                            relative_bounding_box.xmin, relative_bounding_box.ymin, w,
                            h)
                        rect_end_point = _normalized_to_pixel_coordinates(
                            relative_bounding_box.xmin + relative_bounding_box.width,
                            relative_bounding_box.ymin + relative_bounding_box.height, w,
                            h)
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
        cr, cb = find_skinColor_crcb()
        skinImg = crcb_oval(original_frame, cr, cb)

        if len(landmarks_list) == 478:
            upper_x = landmarks_list[10][0]
            upper_y = landmarks_list[10][1]
            lower_x = landmarks_list[152][0]
            lower_y = landmarks_list[152][1]
            if upper_x and upper_y and lower_x and lower_y:
                StraightWarning(warning_message3, upper_x, upper_y, lower_x, lower_y)
            
            # 【Function5】find specific landmarks of face mesh(for ear warning)
            # Right Ear
            rightEarPoints = get_rightEarPoint()
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
            leftEarPoints = get_leftEarPoint()
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
        # 【Function5】create front detection
        with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as front_detection:
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
                        threading.Thread(target=speak, args=("Look Right",)).start()
                    elif y > 10:
                        text = "Look Left"  # Looking Right
                        threading.Thread(target=speak, args=("Look Left",)).start()
                    elif x < -10:
                        text = "Look Up"  # Looking Down
                        threading.Thread(target=speak, args=("Look Up",)).start()
                    elif x > 10:
                        text = "Look Down"  # Looking Up
                        threading.Thread(target=speak, args=("Look Down",)).start()
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
        # display_frame = cv2.add(display_frame, warning_message4)
        cv2.imshow('camera', display_frame)

        # Output picture file
        if keyboard.is_pressed(' '):

            picture = original_frame
            # picture type is numpy.ndarray
            # beauty_face
            if beauty_face_flag:
                picture = beauty_face(picture)
            # segmentation
            if segmentation_flag:
                picture = add_white_bg(picture)
            # light adjustment
            if light_enhance_flag:  # press "l" to use
                # picture = aug(picture)
                Alpha = cv2.getTrackbarPos("Alpha", "Light Adjustment") / 100
                Beta = cv2.getTrackbarPos("Beta", "Light Adjustment")
                picture = light_adjust(picture, Alpha, Beta)

            # resize frame into standard output size
            snap = cutEdge(box_upperLeft_col, box_h, box_w, picture)
            cv2.imshow('Your Headshot', snap)

    # return value of reading the frame (ret) = False
    else:
        print('Ignoring empty camera frame.\n')
        continue

    # close the window -> esc
    if cv2.waitKey(1) == 27:
        break

cap.release()