import cv2
import mediapipe as mp
from PIL import Image
import os
import numpy as np
import time


Down_Image_Path = ['static/Photo/Down/legs.png', 'static/Photo/Down/leggins-1.png']
Top_Image_Path = ['static/Photo/Top/T_Short_White.png', 'static/Photo/Top/T_Short_Red.png', 'static/Photo/Top/T_Short_EnerGym.png', 'static/Photo/Top/Coft-removebg-preview.png', 'static/Photo/Top/top-1.png', 'static/Photo/Top/te.png']




Top_Offset_To_Y = [10, 10, 3, 0, 55, 0]
Top_Offset_To_X = [0, -2, -1, -2, -3, 0]

Top_With = [1050, 1150, 1300, 1280, 400, 1000]
Top_Height = [550, 550, 1000, 820, 300, 500]
Top_HeightMult = [0, 0, 0, 0, 0, 5]


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


def body_Detect(Index, frame, landmarks, offset_Y):
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

    #21/11
    distanceX = int((shoulder_left[0] - shoulder_right[0]) * (Top_With[Index]))
    distanceY = int((hip_left.y - shoulder_left[1]) * int(Top_Height[Index]))



    if distanceX <= 0:
        distanceX *= -1
    ClothYsize, ClothXsize = Top_overlay_image[Index].size

    xSizeForNew = int(distanceX - (distanceX * ProcentSizeX))
    ySizeForNew = int(((xSizeForNew * ClothYsize) / ClothXsize) + ((xSizeForNew * ClothYsize) / ClothXsize) * Top_HeightMult[Index])


    new_size = (max(100, xSizeForNew), max(100, ySizeForNew))
    overlay_image_resized = Top_overlay_image[Index].resize(new_size)


    offsetX = int(((distanceX / 100) * Top_Offset_To_X[Index]) * ProcentSizeXOffSet)
    offsetY = int(((distanceY / 100) * offset_Y) * ProcentSizeYOffSet)


    PosYSholder = shoulder_left[1] * ySize
    center_x = int(((shoulder_left[0] + landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x) / 2) * frame.shape[1]) + offsetX
    center_y = int((PosYSholder + (ySizeForNew / 2))) - offsetY


    overlay_position = (center_x - new_size[0] // 2, center_y - new_size[1] // 2)


    return overlay_image_resized, overlay_position
def leg_Detect(Index, frame, landmarks, offset_Y, withDown):
    ySize, xSize = frame.shape[:2]

    XFSize, YFSize = 640, 480
    ProcentSizeX = (1 - xSize / XFSize)
    ProcentSizeXOffSet = (xSize / XFSize)
    ProcentSizeYOffset = (ySize / YFSize)

    hip_left = np.array([landmarks[mp_pose.PoseLandmark.LEFT_HIP].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP].y])
    hip_right = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y])
    Heel_LeftY = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y


    distanceX = int(((hip_left[0] - hip_right[0]) * (withDown) * 1.2))
    distanceY = int((Heel_LeftY - hip_left[0]) * (Down_Height[Index]))



    if distanceX <= 0:
        distanceX *= -1

    ClothYsize, ClothXsize = Top_overlay_image[Index].size

    xSizeForNew = int(distanceX - (distanceX * ProcentSizeX))
    ySizeForNew = int(((xSizeForNew * ClothYsize) / ClothXsize) + ((xSizeForNew * ClothYsize) / ClothXsize) * 0)

    new_size = (max(100, xSizeForNew), max(100, ySizeForNew))

    overlay_image_resized = Down_overlay_image[Index].resize(new_size)

    offsetY = int((distanceY / 100) * offset_Y * ProcentSizeYOffset)
    offsetX = int((distanceX / 100) * Down_Offset_To_X[Index] * ProcentSizeXOffSet)



    center_x = int((hip_left[0] + landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x) / 2 * frame.shape[1]) + offsetX
    center_y = int(((hip_left[1] + landmarks[mp_pose.PoseLandmark.LEFT_HIP].y) / 2) * frame.shape[0]) - offsetY


    overlay_position = (center_x - new_size[0] // 2, center_y - new_size[1] // 2)


    return overlay_image_resized, overlay_position
def resImage(img, indexT, indexD, offsetTop, offset_Down, with_Down):
    startTime = time.time()
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    image_path = img
    image = cv2.imread(image_path)

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = pose.process(rgb_image)

    if results.pose_landmarks:
        annotated_image = rgb_image.copy()
        landmarks = results.pose_landmarks.landmark


        mp.solutions.drawing_utils.draw_landmarks(annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        annotated_image_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
        if indexT != -1:
            Top_overlay_image_resized, Top_overlay_position= body_Detect(indexT, image, landmarks, Top_Offset_To_Y[indexT] + int(offsetTop))
        Down_overlay_image_resized, Down_overlay_position = leg_Detect(indexD, image, landmarks, Down_Offset_To_Y[indexD] + int(offset_Down), Down_With[indexD] + int(with_Down))

        frame_pil = Image.fromarray(cv2.cvtColor(annotated_image_bgr, cv2.COLOR_BGR2RGB))

        frame_pil.paste(Down_overlay_image_resized, Down_overlay_position, Down_overlay_image_resized)
        if indexT != -1:
            frame_pil.paste(Top_overlay_image_resized, Top_overlay_position, Top_overlay_image_resized)

        annotated_image_bgr = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

        frame_pil = Image.fromarray(cv2.cvtColor(annotated_image_bgr, cv2.COLOR_BGR2RGB))
        if os.path.isfile("PathSave.txt"):
            with open('PathSave.txt', 'r') as file:
                content = file.read()
                if content != "":
                    if os.path.isfile(content):
                        try:
                            os.remove(content)
                        except:
                            print("превышено число запросов")

        path = "static/" + str(str(indexD) + " " + str(indexT) + " " + str(offsetTop) + " " + str(offset_Down) + " " + str(with_Down) + os.path.basename(img))

        with open('PathSave.txt', 'w') as file:
            file.write(path)

        frame_pil.save(path)     
        if os.path.isfile(img):
            try:
                os.remove(img)
            except:
                print()
        print(time.time() - startTime)
        return path
    else:
        os.remove(img)
