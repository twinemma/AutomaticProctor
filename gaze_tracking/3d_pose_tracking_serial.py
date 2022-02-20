import cv2
import numpy as np
import math
import serial
import time
from face_detector import get_face_detector, find_faces
from face_landmarks import get_landmark_model, detect_marks, draw_marks


def isRotationMatrix(R) :
    """
    Check if a matrix is rotation matrix by verify R'* R = I'
    Parameters
    -----------
    R: 3x3 matrix
    Returns
    --------
    true if R is a rotation matrix
    """
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6
 

def rotationMatrixToEulerAngles(R) :
    """
    Calculates rotation matrix to euler angles based on our derived formula
    """

    #sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    sy = math.sqrt(R[2, 1] * R[2, 1] + R[2, 2] * R[2, 2])
    singular = sy < 1e-6
    if  not singular :
        x = math.degrees(math.atan2(R[2,1] , R[2,2]))
        y = math.degrees(math.atan2(-R[2,0], sy))
        z = math.degrees(math.atan2(R[1,0], R[0,0]))
    else :
        print("Encounter singular rotation matrix")
        # TODO: this is suspicous, since sy is 0, then y will be undefined
        x = math.degrees(math.atan2(-R[1,2], R[1,1]))
        y = math.degrees(math.atan2(-R[2,0], sy))
        z = 0
    return np.array([x, y, z])

def eulerAngleFromPnP(rotation_vec, translation_vec):
    # Calculate euler angle
    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    pose_mat = cv2.hconcat((rotation_mat, translation_vec))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
    return (rotation_mat, euler_angles)


def eulerAngleFromRotTrans(rotation_mat, translation_vec):
    # Calculate euler angle
    pose_mat = cv2.hconcat((rotation_mat, translation_vec))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
    return (rotation_mat, euler_angles)


def relativeRotation(R_ca, R_cb) :
    """
    Calculate the relative rotation of object B in the frame of object A.
    Parameters
    -----------
    R_ca: 3x3 rotation matrix of object A in the camera frame (or a transformation from the coordinate frame of
           A to the camera coordinate frame)
    R_cb: 3x3 rotation matrix of object B in the camera frame (or a transformation from the coordinate frame of
           B to the camera coordinate frame)
    Returns
    --------
    R_ab: 3x3 rotation matrix of object B in frame of object A.
    R_ab = R_ac * R_cb = inverse(R_ca) * R_cb = R_ca' * R_cb
    """
    #print("Comparing before {} and after {}".format(R_ca, R_cb))
    relativeMove = np.linalg.norm(R_ca - R_cb)
    R_ac = np.transpose(R_ca)
    return np.dot(R_ac, R_cb)

def withinThreshold(theta_x, theta_y, lb, ub) :
    """
    Check if absolute value of either angle is bigger than the given threshold
    -----------
    theta_x: angle around x-axis
    theta_y: angle around y-axis

    Returns
    --------
    true if either of angle is equal or bigger than the given the threshold
    """
    return (abs(theta_x) >= lb or abs(theta_y) >= lb) and (abs(theta_x) <= ub and abs(theta_y) <= ub)

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging
import tensorflow as tf
tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)  

face_model = get_face_detector()
landmark_model = get_landmark_model()
cap = cv2.VideoCapture(0)
ret, img = cap.read()
size = img.shape
font = cv2.FONT_HERSHEY_SIMPLEX 
#ser = serial.Serial('/dev/cu.usbmodem14101', 115200)
# 3D model points.
model_points = np.array([
                            (0.0, 0.0, 0.0),             # Nose tip
                            (0.0, -330.0, -65.0),        # Chin
                            (-225.0, 170.0, -135.0),     # Left eye left corner
                            (225.0, 170.0, -135.0),      # Right eye right corne
                            (-150.0, -150.0, -125.0),    # Left Mouth corner
                            (150.0, -150.0, -125.0)      # Right mouth corner
                        ])

#Camera internals
focal_length = size[1]
center = (size[1]/2, size[0]/2)
print("camera info: center = {}, focal_length={}".format(center, focal_length))
camera_matrix = np.array(
                         [[focal_length, 0, center[0]],
                         [0, focal_length, center[1]],
                         [0, 0, 1]], dtype = "double"
                         )

prev_rotation = np.zeros((3, 3))
initialized = False
lb = 3 #lower bound
ub = 40 #upper bound
while True:
    ret, img = cap.read()
    if ret == True:
        faces = find_faces(img, face_model)
        for face in faces:
            marks = detect_marks(img, landmark_model, face)
            draw_marks(img, marks, color=(0, 255, 0))
            image_points = np.array([
                                    marks[30],     # Nose tip
                                    marks[8],     # Chin
                                    marks[36],     # Left eye left corner
                                    marks[45],     # Right eye right corne
                                    marks[48],     # Left Mouth corner
                                    marks[54]      # Right mouth corner
                                ], dtype="double")
            dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
            (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
            #(success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_UPNP)
            
            # Project a 3D point (0, 0, 1000.0) onto the image plane.
            # We use this to draw a line sticking out of the nose
            (nose_end_point2D_x, jacobian) = cv2.projectPoints(np.array([(500.0, 0.0, 0.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
            (nose_end_point2D_y, jacobian) = cv2.projectPoints(np.array([(0.0, 500.0, 0.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
            (nose_end_point2D_z, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 500.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
                        
            for p in image_points:
                cv2.circle(img, (int(p[0]), int(p[1])), 3, (0,0,255), -1)
            
            
            p1 = ( int(image_points[0][0]), int(image_points[0][1]))
            px = ( int(nose_end_point2D_x[0][0][0]), int(nose_end_point2D_x[0][0][1]))
            py = ( int(nose_end_point2D_y[0][0][0]), int(nose_end_point2D_y[0][0][1]))
            pz = ( int(nose_end_point2D_z[0][0][0]), int(nose_end_point2D_z[0][0][1]))

            # opencv color is defined as BGR
            #cv2.line(img, p1, px, (255, 0, 0), 2)  #blue 
            #cv2.line(img, p1, py, (0, 255, 0), 2)  #green
            #cv2.line(img, p1, pz, (0, 0, 255), 2)  #red


           
            # Else we need to spend time in calculating euler angle and sending cmd to serial port to control motor
            # calculate relative rotation angles of head pose relative to previous frame
            (rmat, euler_angles) = eulerAngleFromPnP(rotation_vector, translation_vector)
            if not isRotationMatrix(rmat):
                print("ignore an invalid rotation matrix {} from solvePnP after cv2.Rodrigues".format(rmat))
                continue

            # If pose tracking is not started, just do nothing
            if not initialized:
                continue

            # print("Previous position rotation matrix {}".format(prev_rotation))
            # print("Current position rotation matrix {}".format(rmat))
            relative_rotation = relativeRotation(prev_rotation, rmat)
            if not isRotationMatrix(relative_rotation):
                print("Ignore invalid relative rotation matrix in the frame of previous pos {}".format(relative_rotation))
                continue
            [theta_x, theta_y, theta_z] = rotationMatrixToEulerAngles(relative_rotation)
            if (withinThreshold(theta_x, theta_y, lb, ub)):
                print("Sending serial command:  0, {}, {}".format(theta_y, theta_x))
                ser.write(str.encode("0," + str(theta_y) + "," + str(theta_x) + ";"))
                # update prev_rotation with current rotation, ready for next iteration
                prev_rotation = rmat;
 
        cv2.imshow('img', img)

        if cv2.waitKey(1) == ord('a'):
            print("a is pressed, start pose tracking")
            initialized = True
            prev_rotation = rmat
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cv2.destroyAllWindows()
cap.release()