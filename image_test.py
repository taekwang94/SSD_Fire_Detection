
import json
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import torch
from torchvision import models
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F
import torchvision

from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

import os
import json
import re
import random
from math import sqrt
object_save_path = '/disk2/taekwang/fire_dataset/TRAIN_objects.json'
image_save_path = '/disk2/taekwang/fire_dataset/TRAIN_images.json'
with open(object_save_path) as json_file:
    object_json_data = json.load(json_file)

with open(image_save_path) as json_file:
    image_json_data = json.load(json_file)

file_number = 106

#print(object_json_data)

boxlist = object_json_data[file_number]['boxes'][1]
print(object_json_data[file_number])
xmin = boxlist[0]
ymin = boxlist[1]
xmax = boxlist[2]
ymax = boxlist[3]
print(boxlist)
img = Image.open(image_json_data[file_number])
origin_img = img.copy()
draw = ImageDraw.Draw(origin_img)
draw.rectangle(xy=[(xmin,ymin), (xmax,ymax)])

plt.imshow(origin_img)
plt.show()