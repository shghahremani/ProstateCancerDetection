
"""
This python file receives the directories that contains the images, wound masks, and tissue masks.

    :param: root_dir:       Path to the folder containing the three directories
    :param: save_dir:       Path of the folder containing the modified images
    :param: image_folder:   the name of the image folder
    :param: tissue_folder:  the name of the tissue mask folder
    :param: wound_folder:   the name of the wound mask folder
    :return:
"""

import os
from PIL import Image
import json

#
# Defining the required parameters
#

configs=json.load(open("config/data_preprocessing.json"))

root_dir = configs["root_dir"]
save_dir = configs["save_dir"]

image_folder = configs["image_folder"]
wound_folder = configs["wound_folder"]
tissue_folder = configs["tissue_folder"]

model_name= configs['model_name']



myargs = {'min_dim': 300, 'max_dim': 1000, 'new_dim':(300,300)}


def apply_preprocessing(files, img_dir, new_img_dir, pipeline, **kwargs):
    for imgFile in files:
        print("Processing Image " + imgFile + " from folder" + img_dir)
        temp_img = Image.open(img_dir + imgFile)
        for func in pipeline:
            temp_img = func(temp_img, **kwargs)
        temp_img.save(new_img_dir + imgFile)


def limit_image_size(img, min_dim=300.0, max_dim=1000, **kwargs):

    if img.height < min_dim:
        ratio = float(min_dim) / float(img.height)
        img = img.resize((int(img.width*ratio), min_dim) , Image.BICUBIC)

    if img.width > max_dim:
        ratio = float(max_dim) / float(img.width)
        img = img.resize((max_dim, int(img.height*ratio)), Image.ANTIALIAS)

    return img


def rotate_image(img, **kwargs):
    if img.height > img.width:
        img=img.transpose(Image.ROTATE_90)
    return img

def resize_image(img,new_dim=None, **kwargs):
    img = img.resize(new_dim, Image.ANTIALIAS)
    return img

if not os.path.exists(save_dir+image_folder):
    os.makedirs(save_dir+image_folder)

if not wound_folder=="":
    if not os.path.exists(save_dir+wound_folder):
        os.makedirs(save_dir+wound_folder)
if not tissue_folder=="":
    if not os.path.exists(save_dir + tissue_folder):
        os.makedirs(save_dir+tissue_folder)


# Note: the masks have an extention of png


# Choose the pipeline for performing the pre-processing
if model_name=='enet':
    myPipeline = [rotate_image, limit_image_size]
elif model_name == 'mobilenet':
    myPipeline = [resize_image]
elif model_name == 'enet_small':
    myPipeline = [rotate_image, resize_image]
    myargs['new_dim']= (640,480)
else:
    myPipeline =[]

if not image_folder=="":
    image_files = sorted([files for files in os.listdir(root_dir + image_folder) if files.endswith(".jpg")])
    apply_preprocessing(image_files, root_dir + image_folder, save_dir + image_folder, myPipeline, **myargs)
if not wound_folder =="":
    wound_files = sorted([files for files in os.listdir(root_dir + wound_folder) if files.endswith(".png")])
    apply_preprocessing(wound_files, root_dir + wound_folder, save_dir + wound_folder, myPipeline, **myargs)
if not tissue_folder=="":
    tissue_files = sorted([files for files in os.listdir(root_dir + tissue_folder) if files.endswith(".png")])
    apply_preprocessing(tissue_files, root_dir + tissue_folder, save_dir + tissue_folder, myPipeline, **myargs)
