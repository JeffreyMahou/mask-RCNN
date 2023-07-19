## import the necessary packages
from mrcnn.config import Config
from mrcnn import model as modellib

import os
import argparse # parsing tool

from matplotlib import pyplot
from matplotlib.patches import Rectangle

from keras.backend import manual_variable_initialization
manual_variable_initialization(True)

import tensorflow as tf

import tensorflow_datasets as tfds
from mrcnn.visualize import display_instances

parser = argparse.ArgumentParser(description='Computing ')
parser.add_argument('--path', type=str, help='path to the image file', required=True) # path to the image file
args = parser.parse_args()
path = args.path

class myMaskRCNNConfig(Config):
    # give the configuration a recognizable name
    NAME = "MaskRCNN_inference"

    # set the number of GPUs to use along with the number of images
    # per GPU
    IMAGES_PER_GPU = 1

    # number of classes (we would normally add +1 for the background
    # but the background class is *already* included in the class
    # names)
    NUM_CLASSES = 1+80


config = myMaskRCNNConfig()

print("loading  weights for Mask R-CNN model…")
model = modellib.MaskRCNN(mode="inference", config=config, model_dir="./")

tf.keras.Model.load_weights(model.keras_model, os.path.join(path, "mask_rcnn_coco.h5"), by_name=True)

class_names = ["BG", "person", "bicycle", "car", "motorcycle", "airplane",
 "bus", "train", "truck", "boat", "traffic light",
 "fire hydrant", "stop sign", "parking meter", "bench", "bird",
 "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
 "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie",
 "suitcase", "frisbee", "skis", "snowboard", "sports ball",
 "kite", "baseball bat", "baseball glove", "skateboard",
 "surfboard", "tennis racket", "bottle", "wine glass", "cup",
 "fork", "knife", "spoon", "bowl", "banana", "apple",
 "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
 "donut", "cake", "chair", "couch", "potted plant", "bed",
 "dining table", "toilet", "tv", "laptop", "mouse", "remote",
 "keyboard", "cell phone", "microwave", "oven", "toaster",
 "sink", "refrigerator", "book", "clock", "vase", "scissors",
 "teddy bear", "hair drier", "toothbrush"]

# draw an image with detected objects

def draw_image_with_boxes(data, boxes_list):

     # plot the image
     pyplot.imshow(data)
     # get the context for drawing boxes
     ax = pyplot.gca()
     # plot each box
     for box in boxes_list:
          # get coordinates
          y1, x1, y2, x2 = box
          # calculate width and height of the box
          width, height = x2 - x1, y2 - y1
          # create the shape
          rect = Rectangle((x1, y1), width, height, fill=False, color='red', lw=5)
          # draw the box
          ax.add_patch(rect)
     # show the plot
     pyplot.show()

##

dataset = tfds.load('imagenette', split='validation')
ds = dataset.shuffle(1000).take(10)

for image in ds:  # example is (image, label)
     img = image["image"].numpy()
     pyplot.imshow(img)
     pyplot.show()
     # make prediction
     results = model.detect([img], verbose=1)
     # visualize the results
     draw_image_with_boxes(img, results[0]['rois'])

     # get dictionary for first prediction
     r = results[0]
     # show photo with bounding boxes, masks, class labels and scores
     display_instances(img, r["rois"], r["masks"], r["class_ids"], class_names, r["scores"])

     ##

     classes= r['class_ids']
     print("Total Objects found", len(classes))
     for i in range(len(classes)):
         print(class_names[classes[i]])