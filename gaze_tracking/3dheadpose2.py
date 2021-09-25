import cv2
import numpy as np
import math
import serial
from face_detector import get_face_detector, find_faces
from face_landmarks import get_landmark_model, detect_marks
import time

def get_2d_points(img, rotation_vector, translation_vector, camera_matrix, val):
    """Return the 3D points present as 2D for making annotation box"""
    point_3d = []
    dist_coeffs = np.zeros((4,1))
    rear_size = val[0]
    rear_depth = val[1]
    point_3d.append((-rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, -rear_size, rear_depth))
    
    front_size = val[2]
    front_depth = val[3]
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d.append((-front_size, front_size, front_depth))
    point_3d.append((front_size, front_size, front_depth))
    point_3d.append((front_size, -front_size, front_depth))
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d = np.array(point_3d, dtype=np.float).reshape(-1, 3)
    
    # Map to 2d img points
    (point_2d, _) = cv2.projectPoints(point_3d,
                                      rotation_vector,
                                      translation_vector,
                                      camera_matrix,
                                      dist_coeffs)
    point_2d = np.int32(point_2d.reshape(-1, 2))
    return point_2d

def draw_annotation_box(img, rotation_vector, translation_vector, camera_matrix,
                        rear_size=300, rear_depth=0, front_size=500, front_depth=400,
                        color=(255, 255, 0), line_width=2):
    """
    Draw a 3D anotation box on the face for head pose estimation
    Parameters
    ----------
    img : np.unit8
        Original Image.
    rotation_vector : Array of float64
        Rotation Vector obtained from cv2.solvePnP
    translation_vector : Array of float64
        Translation Vector obtained from cv2.solvePnP
    camera_matrix : Array of float64
        The camera matrix
    rear_size : int, optional
        Size of rear box. The default is 300.
    rear_depth : int, optional
        The default is 0.
    front_size : int, optional
        Size of front box. The default is 500.
    front_depth : int, optional
        Front depth. The default is 400.
    color : tuple, optional
        The color with which to draw annotation box. The default is (255, 255, 0).
    line_width : int, optional
        line width of lines drawn. The default is 2.
    Returns
    -------
    None.
    """
    
    rear_size = 1
    rear_depth = 0
    front_size = img.shape[1]
    front_depth = front_size*2
    val = [rear_size, rear_depth, front_size, front_depth]
    point_2d = get_2d_points(img, rotation_vector, translation_vector, camera_matrix, val)
    # # Draw all the lines
    cv2.polylines(img, [point_2d], True, color, line_width, cv2.LINE_AA)
    cv2.line(img, tuple(point_2d[1]), tuple(
        point_2d[6]), color, line_width, cv2.LINE_AA)
    cv2.line(img, tuple(point_2d[2]), tuple(
        point_2d[7]), color, line_width, cv2.LINE_AA)
    cv2.line(img, tuple(point_2d[3]), tuple(
        point_2d[8]), color, line_width, cv2.LINE_AA)
    
    
def head_pose_points(img, rotation_vector, translation_vector, camera_matrix):
    """
    Get the points to estimate head pose sideways    
    Parameters
    ----------
    img : np.unit8
        Original Image.
    rotation_vector : Array of float64
        Rotation Vector obtained from cv2.solvePnP
    translation_vector : Array of float64
        Translation Vector obtained from cv2.solvePnP
    camera_matrix : Array of float64
        The camera matrix
    Returns
    -------
    (x, y) : tuple
        Coordinates of line to estimate head pose
    """
    rear_size = 1
    rear_depth = 0
    front_size = img.shape[1]
    front_depth = front_size*2
    val = [rear_size, rear_depth, front_size, front_depth]
    point_2d = get_2d_points(img, rotation_vector, translation_vector, camera_matrix, val)
    y = (point_2d[5] + point_2d[8])//2
    x = point_2d[2]
    
    return (x, y)

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
    Calculates rotation matrix to euler angles
    The result is the same as MATLAB except the order
    of the euler angles ( x and z are swapped ).
    """

    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        print("Encounter singular rotation matrix")
        # TODO: this is suspicous, since sy is 0, then y will be undefined
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    return np.array([x, y, z])


def yawpitchrolldecomposition(R):
    sin_x    = math.sqrt(R[2,0] * R[2,0] +  R[2,1] * R[2,1])    
    singular  = sin_x < 1e-6
    if not singular:
        z1    = math.atan2(R[2,0], R[2,1])     # around z1-axis
        x     = math.atan2(sin_x,  R[2,2])    # around x-axis
        z2    = math.atan2(R[0,2], -R[1,2])    # around z2-axis
    else: # gimbal lock
        z1    = 0                               # around z1-axis
        x      = math.atan2(sin_x,  R[2,2])     # around x-axis
        z2    = 0                               # around z2-axis

    return 180 * np.array([[z1], [x], [z2]])/math.pi


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
    R_ac = np.transpose(R_ca)
    #print("Rotation {}".format(R_ac))
    return np.dot(R_ac, R_cb)

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

# Camera internals
focal_length = size[1]
center = (size[1]/2, size[0]/2)
camera_matrix = np.array(
                         [[focal_length, 0, center[0]],
                         [0, focal_length, center[1]],
                         [0, 0, 1]], dtype = "double"
                         )

prev_rotation = np.zeros((3, 3))
initialized = False
ts = time.time()
totalY = 0
totalX = 0
while True:
    ret, img = cap.read()
    if ret == True:
        faces = find_faces(img, face_model)
        for face in faces:
            marks = detect_marks(img, landmark_model, face)
            # mark_detector.draw_marks(img, marks, color=(0, 255, 0))
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

            
            # calculate relative rotation angles of head pose relative to previous frame
            rmat = cv2.Rodrigues(rotation_vector)[0]
            if not isRotationMatrix(rmat):
                print("ignore an invalid rotation matrix {} from solvePnP after cv2.Rodrigues".format(rmat))
                continue

            if not initialized:
                prev_rotation = rmat
                initialized = True
                continue
            else:
                relative_rotation = relativeRotation(prev_rotation, rmat)
                if not isRotationMatrix(relative_rotation):
                    print("Ignore invalid relative rotation matrix in the frame of previous pos {}".format(relative_rotation))
                    continue
                [theta_x, theta_y, theta_z] = rotationMatrixToEulerAngles(relative_rotation)
                totalX += theta_x
                totalY += theta_y
                if time.time() - ts >= 5:
                    #[theta_x, theta_y, theta_z] = yawpitchrolldecomposition(relative_rotation)
                    #ser.write(str.encode("0," + str(totalY) + "," + str(totalX) + ";"))
                    #print("Relative rotation euler angles are ({}, {}, {})".format(theta_x, theta_y, theta_z))
                    print("sending eye movement: " + "0," + str(totalY) + "," + str(totalX) + ";")
                    ts = time.time()
                    totalX = 0
                    totalY = 0
                prev_rotation = rmat

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
            cv2.line(img, p1, px, (255, 0, 0), 2)  #blue 
            cv2.line(img, p1, py, (0, 255, 0), 2)  #green
            cv2.line(img, p1, pz, (0, 0, 255), 2)  #red

            
        cv2.imshow('img', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cv2.destroyAllWindows()
cap.release()