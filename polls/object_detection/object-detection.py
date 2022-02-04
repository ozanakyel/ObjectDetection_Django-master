import os
from cv2 import cv2
import numpy as np
import tensorflow as tf
import sys
import time

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

from tensorflow import ConfigProto
from tensorflow import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.8
session = InteractiveSession(config=config)

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'faster_rcnn_inception_v2_coco_2018_01_28'
# select image folder
image_folder_path = r'C:\Users\ozan.akyel\Desktop\project 2021\lumbar\images\valid'
# Grab path to current working directory
CWD_PATH = os.getcwd()
# Number of classes the object detector can identify
NUM_CLASSES = 80

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'data','mscoco_label_map.pbtxt')

# Path to image
# PATH_TO_IMAGE = os.path.join(r'C:\Users\ozan.akyel\Desktop\project 2021\siperlik toka\v4\valid',IMAGE_NAME) ################################# geri alman gerekebilir


# Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')




image = cv2.imread('C:/Users/ozan.akyel/Desktop/1473731-civil-engineering-wallpaper-2560x1440-for-android-tablet.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_expanded = np.expand_dims(image_rgb, axis=0)

# Perform the actual detection by running the model with the image as input
print('####################################')
(boxes, scores, classes, num) = sess.run(
    [detection_boxes, detection_scores, detection_classes, num_detections],
    feed_dict={image_tensor: image_expanded})

# Draw the results of the detection (aka 'visulaize the results')

vis_util.visualize_boxes_and_labels_on_image_array(
    image,
    np.squeeze(boxes),
    np.squeeze(classes).astype(np.int32),
    np.squeeze(scores),
    category_index,
    use_normalized_coordinates=True,
    line_thickness=2,
    min_score_thresh=0.7,
    max_boxes_to_draw=12)

# ipcamera rtsp protocol
# video = cv2.VideoCapture('rtsp://admin:Abc1234*@192.168.1.108/cam/realmonitor?channel=1&subtype=1') # rtsp://admin:Abc1234*@192.168.1.108/cam/realmonitor?channel=1&subtype=1

# ip camera
# video = cv2.VideoCapture('http://admin:Abc1234*@192.168.1.108/cgi-bin/snapshot.cgi?channel=1')

# local webcam
video = cv2.VideoCapture(0)


while True:
    ret, image = video.read()
    zaman = time.time()

    # print(image)
    # PATH_TO_IMAGE = os.path.join(image_folder_path)
    # success, image = video.read()
    # cv2.imshow('234', image)
    # cv2.waitKey(0)
    # print(PATH_TO_IMAGE)
    # y_min = 750  # y yatay
    # x_min = 380  # x dikey
    # distance_y = 800
    # distance_x = 600
    # image = image[x_min:(x_min+distance_x), y_min:(y_min+distance_y)]
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_expanded = np.expand_dims(image_rgb, axis=0)

    # Perform the actual detection by running the model with the image as input
    print('####################################')
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_expanded})

    # Draw the results of the detection (aka 'visulaize the results')

    vis_util.visualize_boxes_and_labels_on_image_array(
        image,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=2,
        min_score_thresh=0.7,
        max_boxes_to_draw=12)

    # All the results have been drawn on image. Now display the image.
    #cv2.imwrite(r".\asd.jpg",image)
    print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
    zaman = time.time()-zaman
    fps = int(1/zaman)
    cv2.putText(image, f"prediction time={round(zaman, 2)} --- fps={fps}",
                (image.shape[0]-300, 20),
                cv2.FONT_HERSHEY_TRIPLEX,
                0.5,
                (0, 255, 0),
                1)
    cv2.imshow('resim', image)
    # time.sleep(1)

    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
# video.release()
cv2.destroyAllWindows()