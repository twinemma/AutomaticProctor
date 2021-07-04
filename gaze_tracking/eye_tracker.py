
#***********************************************************/
#
# Copyright 2021 Emma Li and Brian Li. All rights reserved.
#
#
#**********************************************************/

import cv2
import numpy as np
from face_detector import get_face_detector, find_faces, draw_faces
from face_landmarks import get_landmark_model, detect_marks, draw_marks
import serial
import time
import datetime


def eye_on_mask(mask, side, shape):
    """
    Create ROI on mask of the size of eyes and also find the extreme points of each eye

    Parameters
    ----------
    mask : np.uint8
        Blank mask to draw eyes on
    side : list of int
        the facial landmark numbers of eyes
    shape : Array of uint32
        Facial landmarks

    Returns
    -------
    mask : np.uint8
        Mask with region of interest drawn
    [l, t, r, b] : list
        left, top, right, and bottommost points of ROI

    """
    points = [shape[i] for i in side]
    points = np.array(points, dtype=np.int32)
    mask = cv2.fillConvexPoly(mask, points, 255)
    l = points[0][0]
    t = (points[1][1]+points[2][1])//2
    r = points[3][0]
    b = (points[4][1]+points[5][1])//2
    return mask, [l, t, r, b]

def find_eyeball_position(end_points, cx, cy):
    """Find and return the eyeball positions, i.e. left or right or top or normal"""
    x_ratio = (end_points[0] - cx)/(cx - end_points[2])
    y_ratio = (cy - end_points[1])/(end_points[3] - cy)
    if x_ratio > 3:
        return 1
    elif x_ratio < 0.33:
        return 2
    elif y_ratio < 0.33:
        return 3
    else:
        return 0

    
def contouring(thresh, mid, img, end_points, right=False):
    """
    Find the largest contour on an image divided by a midpoint and subsequently the eye position

    Parameters
    ----------
    thresh : Array of uint8
        Thresholded image of one side containing the eyeball
    mid : int
        The mid point between the eyes
    img : Array of uint8
        Original Image
    end_points : list
        List containing the exteme points of eye
    right : boolean, optional
        Whether calculating for right eye or left eye. The default is False.

    Returns
    -------
    pos: int
        the position where eyeball is:
            0 for normal
            1 for left
            2 for right
            3 for up
    [x, y] : list
        coordinate of the eye ball           

    """
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    try:
        cnt = max(cnts, key = cv2.contourArea)
        M = cv2.moments(cnt)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        if right:
            cx += mid
        cv2.circle(img, (cx, cy), 4, (0, 0, 255), 2)
        pos = find_eyeball_position(end_points, cx, cy)
        return pos, [cx, cy]
    except:
        #pass
        return 0, [0, 0]
    
def process_thresh(thresh):
    """
    Preprocessing the thresholded image

    Parameters
    ----------
    thresh : Array of uint8
        Thresholded image to preprocess

    Returns
    -------
    thresh : Array of uint8
        Processed thresholded image

    """
    thresh = cv2.erode(thresh, None, iterations=2) 
    thresh = cv2.dilate(thresh, None, iterations=4) 
    thresh = cv2.medianBlur(thresh, 3) 
    thresh = cv2.bitwise_not(thresh)
    return thresh


def valid_eye_pos(left, right):
    """
    Check whether detected eye ball position is valid

    Parameters
    ----------
    left : list
        coordinate of left eye ball 
    right : list
        coordinate of right eye ball 
 
    Returns
    -------
    pos: list
        coordinate of eye ball selected 
    """
    if left[0] != -1 and left[1] != -1 :
        return True, left
    elif right[0] != -1 and right[1] != -1 :
        return True, right
    else:
        return False, [-1, -1]

def delta_mov(prev, cur):
    """
    Compute the delta from prev position to current position

    Parameters
    ----------
    prev : list
        coordinate of previous position 
    cur : list
        coordinate of current position 
 
    Returns
    -------
    delta: list
        delta of eye movement 
    """
    # consider reflection in x coordinate
    return [prev[0] - cur[0], cur[1] - prev[1]]


def print_eye_pos(img, left, right):
    """
    Print the side where eye is looking and display on image

    Parameters
    ----------
    img : Array of uint8
        Image to display on
    left : int
        Position obtained of left eye.
    right : int
        Position obtained of right eye.

    Returns
    -------
    None.

    """
    if left == right and left != 0:
        text = ''
        if left == 1:
            print('Looking left')
            text = 'Looking left'
        elif left == 2:
            print('Looking right')
            text = 'Looking right'
        elif left == 3:
            print('Looking up')
            text = 'Looking up'
        font = cv2.FONT_HERSHEY_SIMPLEX 
        cv2.putText(img, text, (30, 30), font,  
                   1, (0, 255, 255), 2, cv2.LINE_AA) 

def process_calibration(curpos):
    global calibration, calib_key_hit, calib_step, calib_center_pos, calib_right_pos, calib_up_pos, calib_dx, calib_dy, start_send_cmd

    if not calib_key_hit:
        return
    calib_key_hit = False
    if calib_step == 0:
        calib_center_pos = curpos
        calib_step = 1
        print("Move eyeball to right side happy face and hit ENTER key...")
    elif calib_step == 1:
        calib_right_pos = curpos
        calib_dx = calib_center_pos[0] - calib_right_pos[0]
        calib_step = 2
        print("Move eyeball to center happy face and hit ENTER key...")
    elif calib_step == 2:
        calib_center_pos = curpos
        calib_step = 3
        print("Move eyeball to upper happy face and hit ENTER key...")
    elif calib_step == 3:
        calib_up_pos = curpos
        calib_dy = calib_up_pos[1] - calib_center_pos[1]
        calib_res = "1," + str(calib_dx) + "," + str(calib_dy) + ";"
        print("Sending to Arduino calibration result: " + calib_res)
        ser.write(str.encode(calib_res))
        print("Move onto Arduino side to calibrate follower-cam...")
        print("After finish follower-cam calibration, look at center happy face and hit ENTER key...")
        calib_step = 4
    elif calib_step == 4:
        print("Your calibration is completed!")
        print("Switching to tracking mode...")
        calibration = False
        start_send_cmd = True
        calib_step = 5
        calib_center_pos = curpos
    else:
        print("Should NOT get in here!!!...")
    


face_model = get_face_detector()
landmark_model = get_landmark_model()
left = [36, 37, 38, 39, 40, 41]
right = [42, 43, 44, 45, 46, 47]

cap = cv2.VideoCapture(0)
ret, img = cap.read()
thresh = img.copy()

cv2.namedWindow('image')
kernel = np.ones((9, 9), np.uint8)

# threshold to determine eye move and stability
epsilon_move = 0 
epsilon_static = 500 
start_move = False
prev_pos = [0, 0]  #initialize current eye position
start_pos = [0, 0]

# threshold to determine head movement and static
epsilon_head_move = 10
epsilon_head_static = 3
prev_head_pos = [0, 0]
start_head_pos = [0, 0]
scale_factor = 1

start_send_cmd = False

# Calibration mode related control vars
calibration = False
calib_step = 0
calib_center_pos = [0,0]
calib_right_pos = [0,0]
calib_up_pos = [0,0]
calib_dx = 0
calib_dy = 0
calib_key_hit = False

# Reset postion to center.
reset_key_pressed = False

key = input("Enter calibration mode (y/n) ?")
if key == "y":
    calibration = True

if calibration:
    print("Entering calibration mode...")
    print("Look at the center happy face and hit ENTER key...")
else:
    print("Entering tracking mode...")
    start_send_cmd = True

# Define the serial port to send command to arduino
ser = serial.Serial('/dev/cu.usbmodem14101', 115200)

def nothing(x):
    pass
cv2.createTrackbar('threshold', 'image', 75, 255, nothing)

while(True):
    ret, img = cap.read()
    rects = find_faces(img, face_model)
    
    for rect in rects:
        
        shape = detect_marks(img, landmark_model, rect)

        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        mask, end_points_left = eye_on_mask(mask, left, shape)
        mask, end_points_right = eye_on_mask(mask, right, shape)
        mask = cv2.dilate(mask, kernel, 5)
        
        eyes = cv2.bitwise_and(img, img, mask=mask)
        mask = (eyes == [0, 0, 0]).all(axis=2)
        eyes[mask] = [255, 255, 255]
        mid = int((shape[42][0] + shape[39][0]) // 2)
        eyes_gray = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY)
        threshold = cv2.getTrackbarPos('threshold', 'image')
        _, thresh = cv2.threshold(eyes_gray, threshold, 255, cv2.THRESH_BINARY)
        thresh = process_thresh(thresh)
        
        eyeball_pos_left, eyeball_l = contouring(thresh[:, 0:mid], mid, img, end_points_left)
        eyeball_pos_right, eyeball_r = contouring(thresh[:, mid:], mid, img, end_points_right, True)

        valid, cur_pos = valid_eye_pos(eyeball_l, eyeball_r)

        if valid:
            if calibration:
                process_calibration(cur_pos)
                continue

            if reset_key_pressed:
                prev_pos = calib_center_pos
                reset_key_pressed = False
                res = "2;"
                print("Sending to Arduino reset signal: " + res)
                ser.write(str.encode(res))

            if prev_pos[0] == 0 and prev_pos[1] == 0:
                print("Prev position is empty...")
                prev_pos = cur_pos

            else:
               
                delta = delta_mov(prev_pos, cur_pos)
                
                if start_send_cmd:
                    ct = datetime.datetime.now()
                    print("Current time: ", ct)
                    ts = ct.timestamp()
                    print("Timestamp: ", ts)
                    ser.write(str.encode("0," + str(delta[0]) + "," + str(delta[1]) + ";"))
                    print("sending eye movement: " + "0," + str(delta[0]) + "," + str(delta[1]) + ";")

                prev_pos = cur_pos
            
        
    cv2.imshow('eyes', img)
    cv2.imshow("image", thresh)
    if calibration:
        if not calib_key_hit and cv2.waitKey(1) & 0xFF== 13:
            calib_key_hit = True
            print("ENTER key is pressed in calibration mode.")
        continue


    if not start_send_cmd and cv2.waitKey(1) == ord('a'):
        print("a is pressed, start sending cmd")
        start_send_cmd = True
    if not reset_key_pressed and cv2.waitKey(1) == ord('r'):
        print("r is pressed, resetting...")
        reset_key_pressed = True
    
cap.release()
cv2.destroyAllWindows()
ser.close()
