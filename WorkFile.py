import cv2
import mediapipe as mp
from PIL import Image
import os
import numpy as np
import time
import gc


Down_Image_Path = ['static/Photo/Down/legs.png', 'static/Photo/Down/leggins-1.png']
Top_Image_Path = ['static/Photo/Top/T_Short_White.png', 'static/Photo/Top/T_Short_Red.png', 'static/Photo/Top/T_Short_EnerGym.png', 'static/Photo/Top/Coft-removebg-preview.png', 'static/Photo/Top/top-1.png', 'static/Photo/Top/te.png']




Top_Offset_To_Y = [10, 10, 3, 0, 55, 0]
Top_Offset_To_X = [0, -2, -1, -2, -3, 0]

Top_With = [1050, 1150, 1300, 1280, 400, 1000]
Top_Height = [550, 550, 1000, 820, 300, 500]
Top_HeightMult = [0, 0, 0.2, 0, 0, 5]


Down_Offset_To_Y = [-30, -10]
Down_Offset_To_X = [0, -1]

Down_With = [3200, 1100]
Down_Height = [600, 450]

Down_overlay_image = []
Top_overlay_image = []



def BeginFun():


    Count_For = 0
    for im in Down_Image_Path:
        Down_overlay_image.append(1)
        Down_overlay_image[Count_For] = Image.open(im)
        Count_For += 1

    Count_For = 0
    for im in Top_Image_Path:
        Top_overlay_image.append(1)
        Top_overlay_image[Count_For] = Image.open(im)
        Count_For += 1


mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)


def body_Detect(Index, frame, landmarks, offset_Y, withTop, offsetXTop):
    ySize, xSize = frame.shape[:2]

    XFSize, YFSize = 640, 480
    ProcentSizeX = (1 - xSize / XFSize)
    ProcentSizeXOffSet = (xSize / XFSize)
    ProcentSizeYOffSet = (ySize / YFSize)

    LeftTop = (landmarks[23].y - landmarks[11].y) / 10
    LeftRight = (landmarks[24].y - landmarks[12].y) / 10

    shoulder_left = np.array([landmarks[11].x, landmarks[11].y - LeftTop])
    shoulder_right = np.array([landmarks[12].x, landmarks[12].y - LeftRight])

    hip_left = landmarks[23]

    distanceX = int((shoulder_left[0] - shoulder_right[0]) * withTop)
    distanceY = int((hip_left.y - shoulder_left[1]) * int(Top_Height[Index]))

    if distanceX <= 0:
        distanceX *= -1

    ClothYsize, ClothXsize = Top_overlay_image[Index].size

    xSizeForNew = int(distanceX - (distanceX * ProcentSizeX))
    ySizeForNew = int(
        ((xSizeForNew * ClothYsize) / ClothXsize) + ((xSizeForNew * ClothYsize) / ClothXsize) * Top_HeightMult[Index])

    new_size = (xSizeForNew, ySizeForNew)

    offsetX = int(((distanceX / 100) * offsetXTop) * ProcentSizeXOffSet)
    offsetY = int(((distanceY / 100) * offset_Y) * ProcentSizeYOffSet)

    PosYSholder = shoulder_left[1] * ySize
    center_x = int(
        ((shoulder_left[0] + landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x) / 2) * frame.shape[1]) + offsetX
    center_y = int((PosYSholder + (ySizeForNew / 2))) - offsetY

    overlay_position = (center_x - new_size[0] // 2, center_y - new_size[1] // 2)

    # Освободить память: удалить ненужные переменные
    del LeftTop, LeftRight, shoulder_left, shoulder_right, hip_left, distanceX, distanceY, ClothYsize, ClothXsize, xSizeForNew, ySizeForNew, offsetX, offsetY, PosYSholder

    return new_size, overlay_position


def leg_Detect(Index, frame, landmarks, offset_Y, withDown, offsetXDown):
    ySize, xSize = frame.shape[:2]

    XFSize, YFSize = 640, 480
    ProcentSizeX = (1 - xSize / XFSize)
    ProcentSizeXOffSet = (xSize / XFSize)
    ProcentSizeYOffset = (ySize / YFSize)
    pose_landmark = mp_pose.PoseLandmark
    hip_left = np.array([landmarks[pose_landmark.LEFT_HIP].x, landmarks[pose_landmark.LEFT_HIP].y])
    hip_right = np.array([landmarks[pose_landmark.RIGHT_HIP].x, landmarks[pose_landmark.RIGHT_HIP].y])
    Heel_LeftY = landmarks[pose_landmark.LEFT_ANKLE].y

    # Освободить память: удалить объект landmarks

    distanceX = int(((hip_left[0] - hip_right[0]) * (withDown) * 1.2))
    distanceY = int((Heel_LeftY - hip_left[0]) * (Down_Height[Index]))

    if distanceX <= 0:
        distanceX *= -1

    ClothYsize, ClothXsize = Top_overlay_image[Index].size

    xSizeForNew = int(distanceX - (distanceX * ProcentSizeX))
    ySizeForNew = int(((xSizeForNew * ClothYsize) / ClothXsize) + ((xSizeForNew * ClothYsize) / ClothXsize) * 0)

    new_size = (max(100, xSizeForNew), max(100, ySizeForNew))

    offsetY = int((distanceY / 100) * offset_Y * ProcentSizeYOffset)
    offsetX = int((distanceX / 100) * offsetXDown * ProcentSizeXOffSet)

    center_x = int((hip_left[0] + landmarks[pose_landmark.RIGHT_HIP].x) / 2 * frame.shape[1]) + offsetX
    center_y = int(((hip_left[1] + landmarks[pose_landmark.LEFT_HIP].y) / 2) * frame.shape[0]) - offsetY

    overlay_position = (center_x - new_size[0] // 2, center_y - new_size[1] // 2)

    # Освободить память: удалить ненужные переменные
    del landmarks, hip_left, hip_right, Heel_LeftY, distanceX, distanceY, ClothYsize, ClothXsize, xSizeForNew, ySizeForNew, offsetY, offsetX, center_x, center_y

    return new_size, overlay_position


def delAll():
    filesRoot = os.listdir("static/")
    for i in filesRoot:
        if i != "Photo" and i != "Input":
            path = "static/" + i
            try:
                os.remove(path)
            except:
                print(i)

def resImage(img, indexT, indexD, offsetTop, offset_Down, with_Down, with_Top, offsetX_Top, offsetDownX, t):
    tN = time.time()
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    image = cv2.imread(img)


    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = pose.process(rgb_image)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
<<<<<<< HEAD
        annotated_image = rgb_image.copy()
=======


>>>>>>> 162b60bfac153ee7aa258aba12079f8d96e3ba73
        annotated_image_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

        with Image.fromarray(cv2.cvtColor(annotated_image_bgr, cv2.COLOR_BGR2RGB)) as frame_pil:
            if indexT != -1:
                TopSize, Top_overlay_position = body_Detect(indexT, image, landmarks,
                                                            Top_Offset_To_Y[indexT] + int(offsetTop),
                                                            Top_With[indexT] + int(with_Top),
                                                            Top_Offset_To_X[indexT] + int(offsetX_Top))
                Top_overlay_image_resized = Top_overlay_image[indexT].resize(TopSize)

            if indexD != -1:
                DownSize, Down_overlay_position = leg_Detect(indexD, image, landmarks,
                                                             Down_Offset_To_Y[indexD] + int(offset_Down),
                                                             Down_With[indexD] + int(with_Down),
                                                             Down_Offset_To_X[indexD] + int(offsetDownX))
                Down_overlay_image_resized = Down_overlay_image[indexD].resize(DownSize)

            if indexD != -1:
                frame_pil.paste(Down_overlay_image_resized, Down_overlay_position, Down_overlay_image_resized)

            if indexT != -1:
                frame_pil.paste(Top_overlay_image_resized, Top_overlay_position, Top_overlay_image_resized)

            path = f"static/{indexD} {indexT} {with_Top} {offsetX_Top} {offsetDownX} {offsetTop} {offset_Down} {with_Down}{os.path.basename(img)}"
            delAll()
            frame_pil.save(path)

            if os.path.isfile(img):
                try:
                    os.remove(img)
                except:
                    print()

        print(time.time() - t)
        print(time.time() - tN)
        pose.close()
        del mp_pose
        del landmarks
        del image
        del rgb_image
        del pose
        del results
        del annotated_image
        del annotated_image_bgr
        del frame_pil

        if indexT != -1:
            del Top_overlay_image_resized, Top_overlay_position, TopSize
        if indexD != -1:
            del Down_overlay_image_resized, Down_overlay_position, DownSize

        gc.collect()

        return path
    else:
        os.remove(img)
        pose.close()
