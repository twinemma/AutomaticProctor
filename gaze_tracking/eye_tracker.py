# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 19:21:18 2020

@author: hp
"""

import cv2
import numpy as np
from face_detector import get_face_detector, find_faces, draw_face
from face_landmarks import get_landmark_model, detect_marks, draw_marks
import serial
import time


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
epsilon_move = 3
epsilon_static = 0.5
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
# Define the serial port to send command to arduino
#ser = serial.Serial('/dev/cu.usbmodem14101', 115200)

def nothing(x):
    pass
cv2.createTrackbar('threshold', 'image', 75, 255, nothing)

while(True):
    #time.sleep(0.5)
    ret, img = cap.read()
    # find face box
    rects = find_faces(img, face_model)
    
    for rect in rects:
        # find face center point to determine head movement
        cur_head_pos = [int((rect[0] + rect[2]) // 2), int((rect[1] + rect[3]) // 2)]
        # find 68 facial landmarks
        shape = detect_marks(img, landmark_model, rect)
        #draw_marks(img, shape)

        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        mask, end_points_left = eye_on_mask(mask, left, shape)
        #draw_face(img, end_points_left) 
        mask, end_points_right = eye_on_mask(mask, right, shape)
        #draw_face(img, end_points_right)
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
            if prev_pos[0] == 0 and prev_pos[1] == 0:
                prev_pos = cur_pos
                prev_head_pos = cur_head_pos
            else:
                #print("prev_pos: " + str(prev_pos[0]) + "," + str(prev_pos[1]))
                #print("cur_pos: " + str(cur_pos[0]) + "," + str(cur_pos[1]))
                delta = delta_mov(prev_pos, cur_pos)
                #print("total delta: " + str(delta[0]) + "," + str(delta[1]))
                #print("delta: " + str(delta[0]) + "," + str(delta[1]))
                if abs(delta[0]) > epsilon_move or abs(delta[1]) > epsilon_move:
                    if not start_move:
                        print("start move")
                        start_move = True
                        start_pos = prev_pos
                        start_head_pos = prev_head_pos
                else:
                    if start_move and abs(delta[0]) < epsilon_static and abs(delta[1]) < epsilon_static :
                        delta = delta_mov(start_pos, cur_pos)
                        delta_head = delta_mov(start_head_pos, cur_head_pos)
                        print("eye movement: " + str(delta[0]) + "," + str(delta[1]))
                        print("head movement: " + str(delta_head[0]) + "," + str(delta_head[1]))
                        if abs(delta_head[0]) < epsilon_head_static and abs(delta_head[1]) < epsilon_head_static:
                            # if head keep still, scale the eye movement vector
                            delta = [delta[0]*scale_factor, delta[1]*scale_factor]
                            print("scaled eye movement: " + str(delta[0]) + "," + str(delta[1]))
                        #stable case
                        # send left eyeball coordinate to serial monitor of Arduino
                        if start_send_cmd:
                            #ser.write(str.encode("1," + str(delta[0]) + "," + str(delta[1]) + ";"))
                            print("sending eye movement: " + "1," + str(delta[0]) + "," + str(delta[1]) + ";")
                        # reset eveything for next movement
                        start_move = False
                prev_pos = cur_pos
                prev_head_pos = cur_head_pos

        #print_eye_pos(img, eyeball_pos_left, eyeball_pos_right)
        # for (x, y) in shape[36:48]:
        #     cv2.circle(img, (x, y), 2, (255, 0, 0), -1)
        
    cv2.imshow('eyes', img)
    cv2.imshow("image", thresh)
    if cv2.waitKey(1) == ord('a'):
        print("a is pressed, start sending cmd")
        start_send_cmd = True
    if cv2.waitKey(1) == ord('b'):
        print("b is pressed, stop sending cmd")
        start_send_cmd = False
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
ser.close()
