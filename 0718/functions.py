import math
import cv2
import numpy as np
import mediapipe as mp
import pyttsx3

from rembg import remove

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_selfie_segmentation = mp.solutions.selfie_segmentation
mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection

size = np.array([(35, 45), (28, 35), (42, 47)])
sizeIndex = 0
s = pyttsx3.init()
s.setProperty('rate', 200)


def empty(v):
    pass


# 定義調整亮度對比的函式
def adjust(i, c, b):
    output = i * (c/100 + 1) - c + b    # 轉換公式
    output = np.clip(output, 0, 255)
    output = np.uint8(output)
    cv2.imshow('Your Headshot', output)


# 定義調整亮度函式
def brightness_fn(val):
    global snap, contrast, brightness
    brightness = val - 100
    adjust(snap, contrast, brightness)


# 定義調整對比度函式
def contrast_fn(val):
    global snap, contrast, brightness
    contrast = val - 100
    adjust(snap, contrast, brightness)


def speak(sentence):
    s.say(sentence)
    s.runAndWait()


def detectBox(height, width, image):
    times = min(math.floor(height / size[sizeIndex][1]), math.floor(width / size[sizeIndex][0]))
    boxWidth = times * size[sizeIndex][0]
    boxHeight = times * size[sizeIndex][1]
    col_upper_left = (width - boxWidth) / 2
    row_upper_left = (height - boxHeight) / 2
    col_bottom_right = (width + boxWidth) / 2
    row_bottom_right = (height + boxHeight) / 2
    # 720*1280
    cv2.rectangle(image, (round(col_upper_left), round(row_upper_left)),
                  (round(col_bottom_right), round(row_bottom_right)), (0, 255, 0), 2)
    return round(col_upper_left), round(row_upper_left), height, boxWidth


def cutEdge(upperLeft, height, width, image):
    cut = image[:height, upperLeft:upperLeft + math.ceil(width)]
    return cut


def OutOfRangeWarning(image, rect_start_point, rect_end_point, box_upperLeft_col, box_upperLeft_row, box_h, box_w):
    warning_message_pos = (40, 60)
    if rect_end_point[0] > (box_upperLeft_col + box_w):
        cv2.putText(image, 'PLEASE MOVE RIGHT', warning_message_pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 5, 4)
    elif rect_end_point[1] > (box_upperLeft_row + box_h):
        cv2.putText(image, 'PLEASE MOVE UP', warning_message_pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 5, 4)
    elif rect_start_point[0] < box_upperLeft_col:
        cv2.putText(image, 'PLEASE MOVE LEFT', warning_message_pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 5, 4)
    elif rect_start_point[1] < box_upperLeft_row:
        cv2.putText(image, 'PLEASE MOVE DOWN', warning_message_pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 5, 4)


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
        # threading.Thread(target=speak, args=("Stay straight",)).start()


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


def light_adjust(img, alpha, beta, h, w):
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


def get_rightEarPoint(landmarks_list):
    rightEar_point = []
    earRight1_x = landmarks_list[127][0] - 8
    earRight1_y = landmarks_list[127][1] - 8
    earRight2_x = landmarks_list[234][0] - 6
    earRight2_y = landmarks_list[234][1] - 6
    earRight3_x = landmarks_list[93][0] - 6
    earRight3_y = landmarks_list[93][1] - 6
    earRight4_x = landmarks_list[132][0] - 6
    earRight4_y = landmarks_list[132][1] - 6
    earRight5_x = landmarks_list[58][0] - 8
    earRight5_y = landmarks_list[58][1] - 8

    earRight_tx1 = round((earRight1_x + earRight2_x * 5) / 6)
    earRight_ty1 = round((earRight1_y + earRight2_y * 5) / 6)
    earRight_tx2 = round((earRight2_x + earRight3_x) / 2)
    earRight_ty2 = round((earRight2_y + earRight3_y) / 2)
    earRight_tx3 = round((earRight3_x + earRight4_x) / 2)
    earRight_ty3 = round((earRight3_y + earRight4_y) / 2)
    earRight_tx4 = round((earRight4_x * 5 + earRight5_x) / 6)
    earRight_ty4 = round((earRight4_y * 5 + earRight5_y) / 6)
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


def get_leftEarPoint(landmarks_list):
    leftEar_point = []
    earLeft1_x = landmarks_list[356][0] + 8
    earLeft1_y = landmarks_list[356][1] + 8
    earLeft2_x = landmarks_list[454][0] + 6
    earLeft2_y = landmarks_list[454][1] + 6
    earLeft3_x = landmarks_list[323][0] + 6
    earLeft3_y = landmarks_list[323][1] + 6
    earLeft4_x = landmarks_list[361][0] + 6
    earLeft4_y = landmarks_list[361][1] + 6

    earLeft_tx1 = round((earLeft1_x + earLeft2_x) / 2)
    earLeft_ty1 = round((earLeft1_y + earLeft2_y) / 2)
    earLeft_tx2 = round((earLeft2_x + earLeft3_x) / 2)
    earLeft_ty2 = round((earLeft2_y + earLeft3_y) / 2)
    earLeft_tx3 = round((earLeft3_x + earLeft4_x) / 2)
    earLeft_ty3 = round((earLeft3_y + earLeft4_y) / 2)
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


def find_skinColor_crcb(landmarks_list, original_frame):
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


def add_white_bg(img, h, w):
    canvas = np.ones((h, w, 3), dtype="uint8")
    canvas *= 255
    person = remove(img)
    for i in range(w):
        canvas[:, i, 0] = canvas[:, i, 0] * (1 - person[:, i, 3] / 255) + person[:, i, 0] * (person[:, i, 3] / 255)
        canvas[:, i, 1] = canvas[:, i, 1] * (1 - person[:, i, 3] / 255) + person[:, i, 1] * (person[:, i, 3] / 255)
        canvas[:, i, 2] = canvas[:, i, 2] * (1 - person[:, i, 3] / 255) + person[:, i, 2] * (person[:, i, 3] / 255)
    return canvas

