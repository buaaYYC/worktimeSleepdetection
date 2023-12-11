import cv2
import numpy as np
import mediapipe as mp
import time
import os

"""
本代码通过检测人脸和人眼确定是否有睡岗的情况
有人脸+闭眼  = 睡岗
无人脸 （趴着或者其他姿势无法检测到人脸）= 睡岗
"""


"""
mp.solutions.face_mesh Mediapipe 库中用于进行人脸关键点检测的模块。
mp_drawing 是我们为这个工具模块创建的一个引用，用于方便地调用其中的绘图函数
"""
mp_facemesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# 包含了从人脸关键点中选择的左眼的一些特定关键点的索引。
chosen_left_eye_idxs = [362, 385, 387, 263, 373, 380]
chosen_right_eye_idxs = [33, 160, 158, 133, 153, 144]
all_chosen_idxs = chosen_left_eye_idxs + chosen_right_eye_idxs

# 初始化MediaPipe人脸检测模块
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.2)

def distance(point_1, point_2):
    """Calculate l2-norm between two points"""
    dist = sum([(i - j) ** 2 for i, j in zip(point_1, point_2)]) ** 0.5
    return dist

"""
get_ear 函数用于计算眼睛的 EAR(Eye Aspect Ratio),这是一种用于衡量眼睛开合状态的指标。EAR 是通过测量眼睛关键点的几何特征来计算的，通常用于检测眼睛的瞬时状态，如眨眼或闭眼。

landmarks: 人脸关键点的列表，其中包含了特定部位（如眼睛）的关键点坐标。
refer_idxs: 用于计算 EAR 的关键点索引列表，通常是表示眼睛的一组关键点。
frame_width: 图像帧的宽度。
frame_height: 图像帧的高度。
"""
def get_ear(landmarks, refer_idxs, frame_width, frame_height):
    try:
        coords_points = []
        for i in refer_idxs:
            lm = landmarks[i]
            coord = mp_drawing._normalized_to_pixel_coordinates(lm.x, lm.y, frame_width, frame_height)
            coords_points.append(coord)

        P2_P6 = distance(coords_points[1], coords_points[5])
        P3_P5 = distance(coords_points[2], coords_points[4])
        P1_P4 = distance(coords_points[0], coords_points[3])

        ear = (P2_P6 + P3_P5) / (2.0 * P1_P4)

    except:
        ear = 0.0
        coords_points = None

    return ear, coords_points

def eye_status_detection(video_path, save_video_path=None):
    cap = cv2.VideoCapture(video_path)

    if save_video_path:
        # 设置保存视频的参数
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(save_video_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (width, height))

    # 设置睡眠判定的阈值和闭眼时间
    sleep_threshold = 0.2  # 根据实际情况调整
    closed_eye_duration_threshold = 0.1  # 闭眼时间阈值，单位：秒

    # 记录闭眼的起始时间
    start_closed_eye_time = None
    start_closed_face_time = None
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        #检测人脸
        results_face = face_detection.process(image)
        if results_face.detections:
            for detection in results_face.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                       int(bboxC.width * iw), int(bboxC.height * ih)
                cv2.rectangle(frame, bbox, (0, 255, 0), 2)
        else:
            if start_closed_face_time == None:
                start_closed_face_time = time.time()
            else:
                # 计算闭眼的持续时间
                closed_face_time = time.time() - start_closed_face_time
                if closed_face_time >= closed_eye_duration_threshold:
                    print("超过{}找不到脸".format(closed_eye_duration_threshold))
                    cv2.putText(frame, "Sleeping", (1, 50), cv2.FONT_HERSHEY_COMPLEX, 0.9, (0, 0, 255), 2)

        image = np.ascontiguousarray(image)
        imgH, imgW, _ = image.shape

        with mp_facemesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
        ) as face_mesh:
            results = face_mesh.process(image)



        if results.multi_face_landmarks:
            for face_id, face_landmarks in enumerate(results.multi_face_landmarks):
                landmarks = face_landmarks.landmark
                left_ear, left_lm_coordinates = get_ear(landmarks, chosen_left_eye_idxs, imgW, imgH)
                right_ear, right_lm_coordinates = get_ear(landmarks, chosen_right_eye_idxs, imgW, imgH)
                avg_ear = (left_ear + right_ear) / 2.0

                # 在图像上显示EAR值
                cv2.putText(frame, f"EAR: {round(avg_ear, 2)}", (1, 24),
                            cv2.FONT_HERSHEY_COMPLEX, 0.9, (255, 255, 255), 2)
                # 判断用户是否在闭眼
                if avg_ear < sleep_threshold:
                    if start_closed_eye_time is None:
                        # 记录闭眼的起始时间
                        start_closed_eye_time = time.time()
                    else:
                        # 计算闭眼的持续时间
                        closed_eye_duration = time.time() - start_closed_eye_time

                        # 如果闭眼时间超过设定的阈值，则触发 "Sleeping" 状态
                        if closed_eye_duration >= closed_eye_duration_threshold:
                            cv2.putText(frame, "Sleeping", (1, 50), cv2.FONT_HERSHEY_COMPLEX, 0.9, (0, 0, 255), 2)
                            print("超过{}检测到闭眼".format(closed_eye_duration_threshold))
                        #睡觉时红色的点标记
                        # 在图像上绘制左眼关键点
                        if(left_lm_coordinates!=None):
                            for coord in left_lm_coordinates:
                                cv2.circle(frame, coord, 1, (0, 0, 255), -1)
                        # 在图像上绘制右眼关键点
                        if(right_lm_coordinates!=None):
                            for coord in right_lm_coordinates:
                                cv2.circle(frame, coord, 1, (0, 0, 255), -1)


                else:
                    # 重置闭眼的起始时间
                    start_closed_eye_time = None
                    cv2.putText(frame, "Not Sleeping", (1, 50), cv2.FONT_HERSHEY_COMPLEX, 0.9, (0, 255, 0), 2)

                    # 不睡觉时绿色的点标记
                    # 在图像上绘制左眼关键点
                    for coord in left_lm_coordinates:
                        cv2.circle(frame, coord, 1, (0, 255, 0), -1)
                    # 在图像上绘制右眼关键点
                    for coord in right_lm_coordinates:
                        cv2.circle(frame, coord, 1, (0, 255, 0), -1)



        cv2.imshow('FaceMesh Detection', frame)
        if save_video_path:
            out.write(frame)  # 将帧写入视频
        # out.write(frame)  # 将帧写入视频
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    if save_video_path:
        out.release()  # 释放 VideoWriter
    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    # 使用示例
    # video_path = "video/sleep.mp4"
    video_path = 0
    save_video_path = "./video/output1.avi"
    eye_status_detection(video_path, save_video_path)