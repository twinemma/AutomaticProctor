import os
import sys

import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.builders import model_builder


############################################################################
# Initialization Phase:                                                    #
#    Load custom-trained model and build in-memory object detection model  #
#                                                                          #
############################################################################
# location to custom trained model
MODELS_DIR = os.path.join(os.getcwd(), 'workspace/training_demo/exported-models')
MODEL_NAME = 'custom_ssd_mobilenet_v2_300x300'
PATH_TO_CKPT = os.path.join(MODELS_DIR, os.path.join(MODEL_NAME, 'checkpoint/'))
PATH_TO_CFG = os.path.join(MODELS_DIR, os.path.join(MODEL_NAME, 'pipeline.config'))
if not os.path.exists(PATH_TO_CKPT):
    sys.exit("Model {} does not exist".format(PATH_TO_CKPT));

# Label Map file
LABEL_FILENAME = 'label_map.pbtxt'
PATH_TO_LABELS = os.path.join(MODELS_DIR, os.path.join(MODEL_NAME, LABEL_FILENAME))
if not os.path.exists(PATH_TO_LABELS):
    sys.exit("Label file {} does not exist".format(PATH_TO_LABELS));


# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Next we load the custom trained model
# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(PATH_TO_CFG)
model_config = configs['model']
detection_model = model_builder.build(model_config=model_config, is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(PATH_TO_CKPT, 'ckpt-0')).expect_partial()


# Load label map data (for plotting)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Label maps correspond index numbers to category names, so that when our convolution network
# predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility
# functions, but anything that returns a dictionary mapping integers to appropriate string labels
# would be fine.
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                    use_display_name=True)

############################################################################
# Configuration Phase:                                                     #
#    Test admin configures cheating targets based on exam policy           #
#                                                                          #
############################################################################
print("*********************************************************************")
print("* For Test Administrators Only                                      *")
print("*                                                                   *")
print("* Cheating Targets Supported:                                       *")
print("* (1) Calculator                                                    *")
print("* (2) Cell Phone                                                    *")
print("* (3) Note                                                          *")
print("* (4) Open Book                                                     *")
print("*********************************************************************")
chosen_target_index = [int(x) for x in input("Please choose your cheating targets separated by comma: ").split(",")]
chosen_target_name = []
chosen_category_index = {}
for item in chosen_target_index:
    chosen_target_name.append(category_index[item]['name'])
    chosen_category_index[item] = category_index[item]
print("You have chosen these cheating targets: {}".format(chosen_target_name))
print("Configured category index: {}".format(chosen_category_index))
cheating_duration = int(input("Please enter cheating duration threshold in seconds: "))
print("You have configued cheating duration threshold as {} seconds".format(cheating_duration))


############################################################################
# Detection Phase:                                                         #
#                                                                          #
# The code shown below loads an image, runs it through the detection model #
# and visualizes the detection results based on configured categories.     #
#                                                                          #
# Note that this will take a long time (several minutes) the first time you#
# run this code due to tf.function's trace-compilation --- on subsequent   #
# runs (e.g. on new images), things will be faster.                        #
############################################################################

# Define the video stream
from urllib.request import urlopen
import cv2

# uncomment below if we are reading from ESP2 camera
# ESP_URL="http://192.168.1.21/cam-hi.jpg"
cap = cv2.VideoCapture(0)


@tf.function
def detect_fn(image):
    """Detect objects in image using initialized detection_model"""

    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)

    return detections, prediction_dict, tf.reshape(shapes, [-1])


def default_format_datetime(ts):
    """Format a datetime with 24 hours format"""
    return format_datetime(ts, '%Y/%m/%d %H:%M:%S')

def format_datetime(ts, format):
    """Format a datetime with the given format"""
    return ts.strftime(format)

def create_video_writer(videoName):
    """Create a video writer to save the video clip"""

    return cv2.VideoWriter(videoName + ".mp4", cv2.VideoWriter_fourcc(*'mp4v'), 20, (800,600))


import numpy as np
from config_vis_util import visualize_boxes_and_labels_on_image_array_by_config
import datetime

# timer set for each class of configured cheating targets
cheating_target_timer_map = {}
# video writer for each class of configured cheating targets for the specified duration window
cheating_target_video_writer_map = {}
while True:
    # Uncomment below if we are reading from ESP2 camera
    # open this URL to trigger ESP32 camera to take an image frame
    # imgResp=urllib.request.urlopen(ESP_URL)
    # imgNp=np.array(bytearray(imgResp.read()),dtype=np.uint8)
    # image_np=cv2.imdecode(imgNp,-1)
    # image_np=cv2.resize(image_np,(640,480))

    # Read frame from camera
    ret, image_np = cap.read()

    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)

    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections, predictions_dict, shapes = detect_fn(input_tensor)
 
    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    detected_target_classes = visualize_boxes_and_labels_on_image_array_by_config(
          image_np_with_detections,
          detections['detection_boxes'][0].numpy(),
          (detections['detection_classes'][0].numpy() + label_id_offset).astype(int),
          detections['detection_scores'][0].numpy(),
          chosen_category_index,
          use_normalized_coordinates=True,
          max_boxes_to_draw=200,
          min_score_thresh=.50,
          agnostic_mode=False)
    # Display output
    displayed_image = cv2.resize(image_np_with_detections, (800, 600))
    cv2.imshow('object detection', displayed_image)

    for target_class in detected_target_classes:
        if target_class in cheating_target_timer_map:
            # this class is already detected before
            cheating_start = cheating_target_timer_map[target_class]
            cheating_end = datetime.datetime.now()
            cheating_delta = cheating_end - cheating_start
            # record the video
            cheating_target_video_writer_map[target_class].write(displayed_image)
            if cheating_delta.seconds > cheating_duration:
                # this class of target has been detected for pre-configured duration threshold
                print("Potential cheating using {} from duration ({}, {})"
                    .format(category_index[target_class]['name'],
                        default_format_datetime(cheating_start),
                        default_format_datetime(cheating_end)))
                cv2.putText(image_np_with_detections, 'Potential Cheating', (30, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
                # stop recording
                cheating_target_video_writer_map[target_class].release()
                # reset its timer and video writer
                del cheating_target_timer_map[target_class]
                del cheating_target_video_writer_map[target_class]
        else:
            # first time detecting this class since last time window,
            # initialize its timer
            cheating_target_timer_map[target_class] = datetime.datetime.now()
            # start video recording
            video_name = "cheating_" + format_datetime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
            video_writer = create_video_writer(video_name)
            cheating_target_video_writer_map[target_class] = video_writer
            video_writer.write(displayed_image)


    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()