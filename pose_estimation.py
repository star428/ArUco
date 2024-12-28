'''
Sample Usage:-
python pose_estimation.py --K_Matrix calibration_matrix.npy --D_Coeff distortion_coefficients.npy --type DICT_5X5_100 -a 127.0.0.1 -p 5000
'''


import threading
import numpy as np
import cv2
import sys
from utils import ARUCO_DICT
import argparse
import time

import json
import queue
import socket

# 创建一个队列来存储要发送的数据
data_queue = queue.Queue()

def udp_sender(address, port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    while True:
        message = data_queue.get()
        if message is None: # 收到 None 时退出
            break
        sock.sendto(message.encode(), (address, port))

def flatten_list(nested_list):
    return [item for sublist in nested_list for item in sublist]

def send_pose_estimation_info(rvec, tvec):
    pose_info = {
        'rvec': flatten_list(rvec.tolist()),
        'tvec': flatten_list(tvec.tolist()),
    }
    message = json.dumps(pose_info)
    print(message)
    data_queue.put(message)

def pose_estimation(frame, aruco_dict_type, matrix_coefficients, distortion_coefficients):
    '''
    frame - Frame from the video stream
    matrix_coefficients - Intrinsic matrix of the calibrated camera
    distortion_coefficients - Distortion coefficients associated with your camera

    return:-
    frame - The frame with the axis drawn on it
    '''

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
    parameters = cv2.aruco.DetectorParameters()

    # Adjust parameters for better accuracy
    parameters.adaptiveThreshWinSizeMin = 3
    parameters.adaptiveThreshWinSizeMax = 23
    parameters.adaptiveThreshWinSizeStep = 10
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    parameters.adaptiveThreshConstant = 7
    parameters.minMarkerPerimeterRate = 0.03
    parameters.maxMarkerPerimeterRate = 4.0
    parameters.polygonalApproxAccuracyRate = 0.03
    parameters.minCornerDistanceRate = 0.05
    parameters.minDistanceToBorder = 3
    parameters.minMarkerDistanceRate = 0.05

    corners, ids, rejected_img_points = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    # If markers are detected
    if len(corners) > 0:
        for i in range(0, len(ids)):
            # Estimate pose of each marker and return the values rvec and tvec
            rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.02, matrix_coefficients, distortion_coefficients)
            # Draw a square around the markers
            cv2.aruco.drawDetectedMarkers(frame, corners)

            # Draw Axis
            cv2.drawFrameAxes(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.01)
            send_pose_estimation_info(rvec, tvec)

    return frame

if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument("-k", "--K_Matrix", required=True, help="Path to calibration matrix (numpy file)")
    ap.add_argument("-d", "--D_Coeff", required=True, help="Path to distortion coefficients (numpy file)")
    ap.add_argument("-t", "--type", type=str, default="DICT_ARUCO_ORIGINAL", help="Type of ArUCo tag to detect")
    ap.add_argument("-a", "--address", required=True, help="IP address to send pose estimation info")
    ap.add_argument("-p", "--port", type=int, required=True, help="Port to send pose estimation info")
    
    args = vars(ap.parse_args())

    
    if ARUCO_DICT.get(args["type"], None) is None:
        print(f"ArUCo tag type '{args['type']}' is not supported")
        sys.exit(0)

    aruco_dict_type = ARUCO_DICT[args["type"]]
    calibration_matrix_path = args["K_Matrix"]
    distortion_coefficients_path = args["D_Coeff"]
    
    address = args["address"]
    port = args["port"]

    # 启动 UDP 发送线程
    sender_thread = threading.Thread(target=udp_sender, args=(address, port))
    sender_thread.start()
    
    k = np.load(calibration_matrix_path)
    d = np.load(distortion_coefficients_path)

    video = cv2.VideoCapture(1)
    time.sleep(2.0)    

    while True:
        ret, frame = video.read()

        if not ret:
            break
        
        output = pose_estimation(frame, aruco_dict_type, k, d)

        cv2.imshow('Estimated Pose', output)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    # 停止 UDP 发送线程
    data_queue.put(None)
    sender_thread.join()
    
    # 释放资源
    video.release()    
    cv2.destroyAllWindows()