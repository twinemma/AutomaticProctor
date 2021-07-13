import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import time
from urllib.request import urlopen
 
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
 
import cv2

# imports from object_detection 
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# downloaded models are in data/models directory
DATA_DIR = os.path.join(os.getcwd(), 'data')
MODELS_DIR = os.path.join(DATA_DIR, 'models')
for dir in [DATA_DIR, MODELS_DIR]:
    if not os.path.exists(dir):
        os.mkdir(dir)

def capture_esp_image(camera_stream, bts):
  """
  function to capture a single image from esp camera stream

  Parameters
  ----------
  camera_stream : opened esp camera stream from URL
  bts: byte buffer

  Returns
  -------
  bts: updated pointer to the byte buffer
  img :Array of uint8 Image that is captured from esp camera. None if no image is captured or exception happens.
  """
  try:
    bts+=camera_stream.read(CAMERA_BUFFRER_SIZE)
    jpghead=bts.find(b'\xff\xd8')
    jpgend=bts.find(b'\xff\xd9')
    if jpghead>-1 and jpgend>-1:
      jpg=bts[jpghead:jpgend+2]
      bts=bts[jpgend+2:]
      img=cv2.imdecode(np.frombuffer(jpg,dtype=np.uint8),cv2.IMREAD_UNCHANGED)
      img=cv2.resize(img,(640,480))
      # h,w=img.shape[:2]
      # print('Resized image Size: height:' + str(h) + 'width：' + str(w))
      return bts, img
    return bts, None
  except Exception as e:
    print("Video capture error:" + str(e))
    return bts, None

def load_graph(trained_model):
  """
  function to load pre-trained frozen model into memory

  Parameters
  ----------
  trained_model : path to pre-trained frozen model file.

  Returns
  -------
  detection_graph : Tensorflow Graph representing the trained model
  """

  frozen_model = os.path.join(MODELS_DIR, trained_model) + '/frozen_inference_graph.pb'
  with tf.compat.v2.io.gfile.GFile(frozen_model, 'rb') as fid:
    od_graph_def = tf.compat.v1.GraphDef()
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)

  with tf.Graph().as_default() as detection_graph:
    tf.import_graph_def(od_graph_def, name='') 
  return detection_graph


def run_inference_for_single_image(image, graph):
  """
  function to run inference for the given image using the TensorFlow graph

  Parameters
  ----------
  image : one image frame.
  graph : TensorFlow Graph

  Returns
  -------
  (boxes, scores, classes, num_detections) : Output tensor values
  """
  # Expand dimensions since the model expects batches of images, which should have shape: [1, xRes, yRes, 3]
  image_expanded = np.expand_dims(image, axis=0)
  image_tensor = graph.get_tensor_by_name('image_tensor:0')
  # Each box represents a part of the image where a particular object was detected.
  boxes = graph.get_tensor_by_name('detection_boxes:0')
  # Each score represent how level of confidence for each of the objects.
  # Score is shown on the result image, together with the class label.
  scores = graph.get_tensor_by_name('detection_scores:0')
  classes = graph.get_tensor_by_name('detection_classes:0')
  num_detections = graph.get_tensor_by_name('num_detections:0')
  # Actual detection by feeding the graph with the expanded image
  return sess.run(
    [boxes, scores, classes, num_detections],
    feed_dict={image_tensor: image_expanded}) 

def download_model_if_not_exists(trained_model):
  """
  Download the trained model if it was not downloaded before. The downlaoded frozen
  model will be saved at the directory named as "trained_model".

  Parameters
  ----------
  trained_model : pre-trained tensorflow model.

  Returns
  -------
  True if the real download is happening.
  """
  dir_to_model = os.path.join(MODELS_DIR, trained_model)
  if not os.path.exists(dir_to_model):
    print("Downloading tensor flow model: " + trained_model)
    model_file = trained_model + '.tar.gz'
    DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
    path_to_model_tar = os.path.join(MODELS_DIR, model_file)
    opener = urllib.request.URLopener()
    opener.retrieve(DOWNLOAD_BASE + model_file, path_to_model_tar)
    tar_file = tarfile.open(path_to_model_tar)
    for file in tar_file.getmembers():
      file_name = os.path.basename(file.name)
      if 'frozen_inference_graph.pb' in file_name:
        tar_file.extract(file, MODELS_DIR) 
    tar_file.close()
    os.remove(path_to_model_tar)
    return True
  return False   

def create_category_index_from_labelmap(label_map, num_classes):
  """
  Load label map and create a category index from label map for classification.

  Parameters
  ----------
  label_map : label map for the pre-trained object detection model.

  Returns
  -------
  Object category dictionary, keyed by 'id' of each category
  """  
  label_map = label_map_util.load_labelmap(label_map)
  categories = label_map_util.convert_label_map_to_categories(label_map, 
    max_num_classes=num_classes, use_display_name=True)
  return label_map_util.create_category_index(categories)

# Model preparation
#MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
#MODEL_NAME = 'ssd_mobilenet_v2_coco_2018_03_29'
MODEL_NAME = 'ssdlite_mobilenet_v2_coco_2018_05_09'
download_model_if_not_exists(MODEL_NAME) 

# Load a (frozen) Tensorflow model into memory. Download the model if not already saved in local
detection_graph = load_graph(MODEL_NAME)

# Loading label map
PATH_TO_LABELS = 'models/research/object_detection/data/mscoco_label_map.pbtxt'
NUM_CLASSES = 90
category_index = create_category_index_from_labelmap(PATH_TO_LABELS, NUM_CLASSES)

# initialize ESP IP Camera stream
ESP_URL="http://192.168.1.19:81/stream"
CAMERA_BUFFRER_SIZE=4096
stream=urlopen(ESP_URL) 
bts=b''

with detection_graph.as_default():
  with tf.compat.v1.Session(graph=detection_graph) as sess:
    while True:
      start_time = time.time()
      bts, image = capture_esp_image(stream, bts)
      if image is not None:
        (boxes, scores, classes, num_detections) = run_inference_for_single_image(image, detection_graph)
        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
            image,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8)
        print ("---receive and processing frame time: {} seconds ---".format(time.time() - start_time))    
        cv2.imshow('object detection', cv2.resize(image, (1600,1200)))
      if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break