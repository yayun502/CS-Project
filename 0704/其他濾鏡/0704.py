import cv2
import dlib
import numpy as np

cap = cv2.VideoCapture(0)


# 得到圖片中的人臉關鍵點
# 輸入參數
# img ： 圖片
# det_face ： 人臉檢測器
# det_landmarks ： 人臉關鍵點檢測器
def get_landmarks_points(img, det_face, det_landmarks):
    # 轉換成灰階圖片
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 檢測人臉區域
    face_rects = det_face(gray, 0)

    # 得到dlib的68個關鍵點
    landmarks = det_landmarks(gray, face_rects[0])

    # 得到關鍵點的坐標
    landmarks_points = []
    parts = landmarks.parts()
    for part in parts:
        landmarks_points.append((part.x, part.y))
    return landmarks_points


# 得到小三角形們的頂點list
# landmarks_points ： 68個關鍵點的坐標
def get_tri_pt_index_list(landmarks_points):
    points = np.array(landmarks_points, np.int32)

    # 得到人臉區域的convex
    convexhull = cv2.convexHull(points)

    # 得到該convex的外接矩形
    rect = cv2.boundingRect(convexhull)

    # 利用subdiv進行三角剖分
    subdiv = cv2.Subdiv2D(rect)
    subdiv.insert(landmarks_points)
    triangles = subdiv.getTriangleList()

    triangles = np.array(triangles, dtype=np.int32)

    # 得到每個三角形的坐標位置
    list_index_tris = []
    for t in triangles:
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])

        index_pt1 = np.where((points == pt1).all(axis=1))[0]
        index_pt2 = np.where((points == pt2).all(axis=1))[0]
        index_pt3 = np.where((points == pt3).all(axis=1))[0]

        if index_pt1.size != 0 and index_pt2.size != 0 and index_pt3.size != 0:
            list_index_tris.append((index_pt1[0], index_pt2[0], index_pt3[0]))
    return list_index_tris


def get_one_rect_from_tri(img, tri, landmarks):
    # 得到三個頂點的坐標'
    pt1 = landmarks[tri[0]]
    pt2 = landmarks[tri[1]]
    pt3 = landmarks[tri[2]]
    points = np.array([pt1, pt2, pt3], dtype=np.int32)

    # 做一个外接矩形
    crop_rect = cv2.boundingRect(points)
    (x, y, w, h) = crop_rect

    # 計算三個頂點在外接矩形上的坐標((0,0)為起始點)
    points_in_rect = points - np.array([(x, y)])

    # 擷取該圖片
    crop_img = img[y:y + h, x:x + w]

    return crop_img, crop_rect, points_in_rect


def get_face_cover(img_src, img_dst, landmarks_src, landmarks_dst, list_index_tris):
    img_cover = np.zeros_like(img_dst, np.uint8)

    for tri in list_index_tris:
        # source image的三角形上截取一個矩形(才能做後續轉換，因為三角形不行相互轉換)
        crop_img_src, crop_rect_src, points_in_rect_src = get_one_rect_from_tri(img_src, tri, landmarks_src)

        # 從destination image的相同位置上也截取一個矩形
        crop_img_dst, crop_rect_dst, points_in_rect_dst = get_one_rect_from_tri(img_dst, tri, landmarks_dst)

        # 計算tranform matrix
        pts_src = np.float32(points_in_rect_src)
        pts_dst = np.float32(points_in_rect_dst)
        M = cv2.getAffineTransform(pts_src, pts_dst)

        # 進行轉換(source to destination)
        (x, y, w, h) = crop_rect_dst
        warped_src = cv2.warpAffine(crop_img_src, M, (w, h))

        # 因為只需替換圖片中三角形區域
        # 因此需要建立一個mask, 對三角形區域裡填充值255，其他部分填充值為0
        mask_dst = np.zeros((h, w), np.uint8)
        cv2.fillConvexPoly(mask_dst, points_in_rect_dst, 255)
        warped_tri = cv2.bitwise_and(warped_src, warped_src, mask=mask_dst)

        # 直接疊加三角形，則連接處會重複相加，所以在疊加時，只針對區域內值非零的部分進行疊加
        # img_mask[y:y+h,x:x+w] = img_mask[y:y+h,x:x+w] + warped_tri
        img_area = img_cover[y:y + h, x:x + w].copy()
        img_area_gray = cv2.cvtColor(img_area, cv2.COLOR_BGR2GRAY)
        _, mask_area = cv2.threshold(img_area_gray, 1, 255, cv2.THRESH_BINARY_INV)
        warped_tri = cv2.bitwise_and(warped_tri, warped_tri, mask=mask_area)
        img_area = cv2.add(img_area, warped_tri)
        img_cover[y:y + h, x:x + w] = img_area

    return img_cover


def face_swap(img_dst, img_cover, landmarks_dst):
    # 得到人臉部分的convex
    img_dst_gray = cv2.cvtColor(img_dst, cv2.COLOR_BGR2GRAY)
    points = np.array(landmarks_dst, np.int32)
    convexhull = cv2.convexHull(points)

    # convex填滿255 得到mask以獲取非人臉部分
    face_mask = np.zeros_like(img_dst_gray)
    face_mask_255 = cv2.fillConvexPoly(face_mask, convexhull, 255)
    face_mask_0 = cv2.bitwise_not(face_mask_255)
    img_noface = cv2.bitwise_and(img_dst, img_dst, mask=face_mask_0)

    # 將非人臉 和 人臉部分 疊加
    result = cv2.add(img_noface, img_cover)
    # cv2.imshow("Image_result", result)

    # 顏色調整
    (x, y, w, h) = cv2.boundingRect(convexhull)
    center_face = (int((x + x + w) / 2), int((y + y + h) / 2))

    seamlessclone = cv2.seamlessClone(result, img_dst, face_mask_255, center_face, cv2.NORMAL_CLONE)

    return seamlessclone


while True:

    ret, frame = cap.read()

    if ret:
        # 建立人臉檢測器
        det_face = dlib.get_frontal_face_detector()

        # 載入標誌點檢測器
        det_landmarks = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # 68点

        # 載入圖片
        # img_dst = cv2.imread('face/sunhonglei.jpg')
        # img_src = cv2.imread('face/baijingting.jpg')

        # img_src = cv2.imread('face/angelababy.png')
        img_src = cv2.imread('face/baijingting.png')
        img_src = cv2.resize(img_src, (0, 0), fx=0.5, fy=0.5)
        img_dst = frame

        # 得到source image的68個關鍵點坐標
        landmarks_src = get_landmarks_points(img_src, det_face, det_landmarks)

        # 得到destination image的68個關鍵點坐標
        landmarks_dst = get_landmarks_points(img_dst, det_face, det_landmarks)

        # 得到用來進行三角剖分的關鍵點的index list
        list_index_tris = get_tri_pt_index_list(landmarks_src)

        # 得到destination image中 需要替換的部分
        img_cover = get_face_cover(img_src, img_dst, landmarks_src, landmarks_dst, list_index_tris)

        # 進行人臉替換
        result = face_swap(img_dst, img_cover, landmarks_dst)

        # cv2.imshow("img src", img_src)
        # cv2.imshow("img dst", img_dst)
        # cv2.imshow("img cover", img_cover)
        cv2.imshow("result", result)

        # return value of reading the frame (ret) = False
    else:
        print('Ignoring empty camera frame.\n')
        continue

    # close the window -> esc
    if cv2.waitKey(1) == 27:
        break

cap.release()
