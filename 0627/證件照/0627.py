import math
import cv2
import numpy as np
import keyboard
import mediapipe as mp
import faceBlendCommon as fbc
import csv

from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates
from rembg import remove
from scipy.constants import sigma

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_selfie_segmentation = mp.solutions.selfie_segmentation
mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection

cap = cv2.VideoCapture(0)
size = np.array([(35, 45), (28, 35), (42, 47)])
sizeIndex = 0
light_enhance_flag = False
beauty_face_flag = False
segmentation_flag = True
BG_COLOR = (255, 255, 255)  # white
Alpha = 1.8
Beta = 10

VISUALIZE_FACE_POINTS = False
filters_config = {
    'spider':
        [{'path': "mask/spider.png",
          'anno_path': "annotation/spider_annotations.csv",
          'morph': True, 'animated': False, 'has_alpha': True
        }]
}

def getLandmarks(img):
    mp_face_mesh = mp.solutions.face_mesh
    selected_keypoint_indices = [127, 93, 58, 136, 150, 149, 176, 148, 152, 377, 40, 285, 296, 334, 293, 300, 168, 6, 195, 4, 64, 60, 94, 290, 439, 33, 387,466, 373, 380, 61, 40, 39, 0, 269, 270, 291, 321, 405, 17, 18, 178, 162, 54, 67, 10, 297,284, 389]
    with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True,
                               min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
        # Convert the BGR frame to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # To improve performance, mark the frame as not writeable to pass by reference.
        img.flags.writeable = False
        result = face_mesh.process(img)
        img.flags.writeable = True
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        if result.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                values = np.array(face_landmarks.landmark)
                face_keypnts = np.zeros((len(values), 2))
                for idx, value in enumerate(values):
                    face_keypnts[idx][0] = value.x
                    face_keypnts[idx][1] = value.y

                face_keypnts = face_keypnts * (w, h)
                face_keypnts = face_keypnts.astype('int')

                relevent_keypnts = []

                for i in selected_keypoint_indices:
                    relevent_keypnts.append(face_keypnts[i])
                return relevent_keypnts
    return 0


def load_filter_img(img_path, has_alpha):
    # Read the image
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

    alpha = None
    if has_alpha:
        b, g, r, alpha = cv2.split(img)
        img = cv2.merge((b, g, r))

    return img, alpha


def load_landmarks(annotation_file):
    with open(annotation_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        points = {
    }
        for i, row in enumerate(csv_reader):
            # skip head or empty line if it's there
            try:
                x, y = int(row[1]), int(row[2])
                points[row[0]] = (x, y)
            except ValueError:
                continue
        return points


def find_convex_hull(points):
    hull = []
    hullIndex = cv2.convexHull(np.array(list(points.values())), clockwise=False, returnPoints=False)
    addPoints = [
        [48], [49], [50], [51], [52], [53], [54], [55], [56], [57], [58], [59],  # Outer lips
        [60], [61], [62], [63], [64], [65], [66], [67],  # Inner lips
        [27], [28], [29], [30], [31], [32], [33], [34], [35],  # Nose
        [36], [37], [38], [39], [40], [41], [42], [43], [44], [45], [46], [47],  # Eyes
        [17], [18], [19], [20], [21], [22], [23], [24], [25], [26]  # Eyebrows
    ]
    hullIndex = np.concatenate((hullIndex, addPoints))
    for i in range(0, len(hullIndex)):
        hull.append(points[str(hullIndex[i][0])])

    return hull, hullIndex


def load_filter(filter_name="spider"):

    filters = filters_config[filter_name]

    multi_filter_runtime = []

    for filter in filters:
        temp_dict = {}

        img1, img1_alpha = load_filter_img(filter['path'], filter['has_alpha'])

        temp_dict['img'] = img1
        temp_dict['img_a'] = img1_alpha

        points = load_landmarks(filter['anno_path'])

        temp_dict['points'] = points

        if filter['morph']:
            # Find convex hull for delaunay triangulation using the landmark points
            hull, hullIndex = find_convex_hull(points)

            # Find Delaunay triangulation for convex hull points
            sizeImg1 = img1.shape
            rect = (0, 0, sizeImg1[1], sizeImg1[0])
            dt = fbc.calculateDelaunayTriangles(rect, hull)

            temp_dict['hull'] = hull
            temp_dict['hullIndex'] = hullIndex
            temp_dict['dt'] = dt

            if len(dt) == 0:
                continue

        if filter['animated']:
            filter_cap = cv2.VideoCapture(filter['path'])
            temp_dict['cap'] = filter_cap

        multi_filter_runtime.append(temp_dict)

    return filters, multi_filter_runtime


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


def beauty_face(img):
    v1 = 3
    v2 = 1
    dx = v1 * 5  # ????????????????????????
    fc = v1 * 12.5  # ????????????????????????
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
    x = img.shape
    c = x[1]
    r = x[0]
    arr = np.array(img)
    for i in range(r):
        for j in range(c):
            for k in range(3):
                temp = arr[i][j][k] * alpha + beta
                if temp > 255:
                    arr[i][j][k] = 2 * 255 - temp
                else:
                    arr[i][j][k] = temp
    return arr


def crcb_oval(image, average_skinColor_cr, average_skinColor_cb):
    """
    YCrCb??????????????? Cr Cb??????
    ??????:??????frame?????????????????????????????????????????????????????????????????????????????????
    ??????:?????????(?????????????????????????????????)
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
    # YCrCb???????????????Cr?????? + otsu?????????:??????????????????threshold???
    # ??????:?????????
    # ??????:??????????????????/????????????????????????
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
    """????????????????????????????????????1?????????????????????????????????"""
    max_percentile_pixel = np.percentile(img, max_percentile)  # param 2?????????????????????????????????
    min_percentile_pixel = np.percentile(img, min_percentile)

    return max_percentile_pixel, min_percentile_pixel


def aug(src):
    """??????????????????"""
    if get_lightness(src) > 130:
        print("?????????????????????????????????")
    # ?????????????????????????????????????????????????????????????????????????????????????????????
    # ??????1????????????????????????0???255????????????????????????????????????????????????0???20??????
    max_percentile_pixel, min_percentile_pixel = compute(src, 1, 99)

    # ?????????????????????????????????
    src[src >= max_percentile_pixel] = max_percentile_pixel
    src[src <= min_percentile_pixel] = min_percentile_pixel

    # ???????????????????????????0???255???????????????255*0.1???255*0.9??????????????????????????????????????????????????????????????????????????????0???255???
    out = np.zeros(src.shape, src.dtype)
    cv2.normalize(src, out, 255 * 0.1, 255 * 0.9, cv2.NORM_MINMAX)

    return out


def get_lightness(src):
    # ????????????
    hsv_image = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    lightness = hsv_image[:, :, 2].mean()

    return lightness


def get_rightEarPoint():
    rightEar_point = []
    earRight1_real_x = round(face_landmarks.landmark[127].x * w - 8)
    earRight1_real_y = round(face_landmarks.landmark[127].y * h - 8)
    earRight2_real_x = round(face_landmarks.landmark[234].x * w - 6)
    earRight2_real_y = round(face_landmarks.landmark[234].y * h - 6)
    earRight3_real_x = round(face_landmarks.landmark[93].x * w - 6)
    earRight3_real_y = round(face_landmarks.landmark[93].y * h - 6)
    earRight4_real_x = round(face_landmarks.landmark[132].x * w - 6)
    earRight4_real_y = round(face_landmarks.landmark[132].y * h - 6)
    earRight5_real_x = round(face_landmarks.landmark[58].x * w - 8)
    earRight5_real_y = round(face_landmarks.landmark[58].y * h - 8)

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
    earLeft1_real_x = round(face_landmarks.landmark[356].x * w + 8)
    earLeft1_real_y = round(face_landmarks.landmark[356].y * h + 8)
    earLeft2_real_x = round(face_landmarks.landmark[454].x * w + 6)
    earLeft2_real_y = round(face_landmarks.landmark[454].y * h + 6)
    earLeft3_real_x = round(face_landmarks.landmark[323].x * w + 6)
    earLeft3_real_y = round(face_landmarks.landmark[323].y * h + 6)
    earLeft4_real_x = round(face_landmarks.landmark[361].x * w + 6)
    earLeft4_real_y = round(face_landmarks.landmark[361].y * h + 6)

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
    with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True,
                               min_detection_confidence=0.5, min_tracking_confidence=0.5) as Face_mesh:
        tmp = original_frame
        tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB)
        tmp.flags.writeable = False
        Results = Face_mesh.process(tmp)
        tmp.flags.writeable = True
        tmp = cv2.cvtColor(tmp, cv2.COLOR_RGB2BGR)

        if Results.multi_face_landmarks:
            for face_landmarks in Results.multi_face_landmarks:
                upper_left_x = round(face_landmarks.landmark[340].x * w)
                upper_left_y = round(face_landmarks.landmark[340].y * h)
                lower_left_x = round(face_landmarks.landmark[322].x * w)
                lower_left_y = round(face_landmarks.landmark[322].y * h)
                upper_right_x = round(face_landmarks.landmark[120].x * w)
                upper_right_y = round(face_landmarks.landmark[120].y * h)
                lower_right_x = round(face_landmarks.landmark[215].x * w)
                lower_right_y = round(face_landmarks.landmark[215].y * h)
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
    h = img.shape[0]
    w = img.shape[1]
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
    if keyboard.is_pressed('up'):
        Beta += 5
        print("Beta is modified larger from ", Beta - 10, " to ", Beta)
    if keyboard.is_pressed('down'):
        Beta -= 5
        print("Beta is modified smaller from ", Beta + 10, " to ", Beta)

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
        # get skin color image
        cr, cb = find_skinColor_crcb()
        skinImg = crcb_oval(original_frame, cr, cb)

        # ???Function2-1???create face detection
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

        # ???Function3-1??? get edge of head image
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

            # ???Function3-2??? find top position of the head
            top_position = None
            for row in range(h):
                for column in range(w):
                    if canny[row][column] == 255:
                        top_position = [row, column]
                        break
                if top_position is not None:
                    break
        # ???Function3-3??? get edge of head image
        if rect_start_point and rect_end_point:
            DistanceWarning(warning_message2, rect_start_point, rect_end_point, top_position, box_h)

        # ???Function4???create face mesh (for straight direction of head)
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
                for face_landmarks in results.multi_face_landmarks:
                    upper_x = round(face_landmarks.landmark[10].x * w)
                    upper_y = round(face_landmarks.landmark[10].y * h)
                    lower_x = round(face_landmarks.landmark[152].x * w)
                    lower_y = round(face_landmarks.landmark[152].y * h)

                    if upper_x and upper_y and lower_x and lower_y:
                        StraightWarning(warning_message3, upper_x, upper_y, lower_x, lower_y)

                    # ???Function5???find specific landmarks of face mesh(for ear warning)
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
                    '''
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

        final_dynamic_frame = frame

        # Displaying
        flipped_frame = cv2.flip(final_dynamic_frame, 1)
        display_frame = cv2.add(flipped_frame, warning_message1)
        display_frame = cv2.add(display_frame, warning_message2)
        display_frame = cv2.add(display_frame, warning_message3)
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
                picture = aug(picture)

            # resize frame into standard output size
            snap = cutEdge(box_upperLeft_col, box_h, box_w, picture)
            cv2.imshow('???????????????', snap)
            # cv2.imshow('adjustment', picture)
            # cv2.imshow('original', original_frame)


        ###########################################################################
        points2 = getLandmarks(original_frame)
        # if face is partially detected
        if not points2 or (len(points2) != 75):
            continue
        # Optical Flow and Stabilization Code
        img2Gray = cv2.cvtColor(original_frame, cv2.COLOR_BGR2GRAY)

        lk_params = dict(winSize=(101, 101), maxLevel=15, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.001))
        points2Next, st, err = cv2.calcOpticalFlowPyrLK(img2GrayPrev, img2Gray, points2Prev, np.array(points2, np.float32), **lk_params)

        # Final landmark points are a weighted average of detected landmarks and tracked landmarks

        for k in range(0, len(points2)):
            d = cv2.norm(np.array(points2[k]) - points2Next[k])
            alpha = math.exp(-d * d / sigma)
            points2[k] = (1 - alpha) * np.array(points2[k]) + alpha * points2Next[k]
            points2[k] = fbc.constrainPoint(points2[k], frame.shape[1], frame.shape[0])
            points2[k] = (int(points2[k][0]), int(points2[k][1]))

        # Update variables for next pass
        points2Prev = np.array(points2, np.float32)
        img2GrayPrev = img2Gray
        ################ End of Optical Flow and Stabilization Code ###############

        if VISUALIZE_FACE_POINTS:
            for idx, point in enumerate(points2):
                cv2.circle(frame, point, 2, (255, 0, 0), -1)
                cv2.putText(frame, str(idx), point, cv2.FONT_HERSHEY_SIMPLEX, .3, (255, 255, 255), 1)
            # cv2.imshow("landmarks", frame)

        filters, multi_filter_runtime = load_filter("spider")

        for idx, filter in enumerate(filters):

            filter_runtime = multi_filter_runtime[idx]
            img1 = filter_runtime['img']
            points1 = filter_runtime['points']
            img1_alpha = filter_runtime['img_a']

            if filter['morph']:

                hullIndex = filter_runtime['hullIndex']
                dt = filter_runtime['dt']
                hull1 = filter_runtime['hull']

                # create copy of frame
                warped_img = np.copy(frame)

                # Find convex hull
                hull2 = []
                for i in range(0, len(hullIndex)):
                    hull2.append(points2[hullIndex[i][0]])

                mask1 = np.zeros((warped_img.shape[0], warped_img.shape[1]), dtype=np.float32)
                mask1 = cv2.merge((mask1, mask1, mask1))
                img1_alpha_mask = cv2.merge((img1_alpha, img1_alpha, img1_alpha))

                # Warp the triangles
                for i in range(0, len(dt)):
                    t1 = []
                    t2 = []

                    for j in range(0, 3):
                        t1.append(hull1[dt[i][j]])
                        t2.append(hull2[dt[i][j]])

                    fbc.warpTriangle(img1, warped_img, t1, t2)
                    fbc.warpTriangle(img1_alpha_mask, mask1, t1, t2)

                # Blur the mask before blending
                mask1 = cv2.GaussianBlur(mask1, (3, 3), 10)

                mask2 = (255.0, 255.0, 255.0) - mask1

                # Perform alpha blending of the two images
                temp1 = np.multiply(warped_img, (mask1 * (1.0 / 255)))
                temp2 = np.multiply(frame, (mask2 * (1.0 / 255)))
                output = temp1 + temp2
            else:
                dst_points = [points2[int(list(points1.keys())[0])], points2[int(list(points1.keys())[1])]]
                tform = fbc.similarityTransform(list(points1.values()), dst_points)
                # Apply similarity transform to input image
                trans_img = cv2.warpAffine(img1, tform, (frame.shape[1], frame.shape[0]))
                trans_alpha = cv2.warpAffine(img1_alpha, tform, (frame.shape[1], frame.shape[0]))
                mask1 = cv2.merge((trans_alpha, trans_alpha, trans_alpha))

                # Blur the mask before blending
                mask1 = cv2.GaussianBlur(mask1, (3, 3), 10)

                mask2 = (255.0, 255.0, 255.0) - mask1

                # Perform alpha blending of the two images
                temp1 = np.multiply(trans_img, (mask1 * (1.0 / 255)))
                temp2 = np.multiply(frame, (mask2 * (1.0 / 255)))
                output = temp1 + temp2

            frame = output = np.uint8(output)

        cv2.imshow("Face Filter", output)


    # return value of reading the frame (ret) = False
    else:
        print('Ignoring empty camera frame.\n')
        continue

    # close the window -> esc
    if cv2.waitKey(1) == 27:
        break

cap.release()
