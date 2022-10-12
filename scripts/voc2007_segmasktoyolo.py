import cv2
import PIL.Image as Image 
import numpy as np 

import glob
import os
import json

VOC_CLASSES = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "potted plant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]


VOC_COLORMAP = [
    [0, 0, 0],
    [128, 0, 0],
    [0, 128, 0],
    [128, 128, 0],
    [0, 0, 128],
    [128, 0, 128],
    [0, 128, 128],
    [128, 128, 128],
    [64, 0, 0],
    [192, 0, 0],
    [64, 128, 0],
    [192, 128, 0],
    [64, 0, 128],
    [192, 0, 128],
    [64, 128, 128],
    [192, 128, 128],
    [0, 64, 0],
    [128, 64, 0],
    [0, 192, 0],
    [128, 192, 0],
    [0, 64, 128],
]


def _convert_to_segmentation_mask(mask):
    # This function converts a mask from the Pascal VOC format to the format required by AutoAlbument.
    # Pascal VOC uses an RGB image to encode the segmentation mask for that image. RGB values of a pixel
    # encode the pixel's class.
    height, width = mask.shape[:2]
    segmentation_mask_dict = {}
    for label_index, label in enumerate(VOC_COLORMAP):
        if label_index == 0:
            continue 
        curr_mask = np.all(mask == label, axis=-1).astype(float)
        if np.count_nonzero(curr_mask) > 0:
            segmentation_mask_dict[VOC_CLASSES[label_index]] = curr_mask
    return segmentation_mask_dict
    
def binarise_mask(mask_fname):
    mask = Image.open(mask_fname)
    mask = np.array(mask)
    # instances are encoded as different colors
    obj_ids = np.unique(mask)
    # first id is the background, so remove it
    obj_ids = obj_ids[1:]
    # Remove the white bundary
    boundary = obj_ids == 255
    obj_ids = obj_ids[~boundary]
    obj_id_to_mask = {}
    for obj_id_index, obj_id in enumerate(obj_ids):
        # split the color-encoded mask into a set of binary masks
        obj_id_to_mask[obj_id] = mask == obj_ids[obj_id_index, None, None]
    
    # mask = cv2.imread(mask_fname)
    # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    # mask = _convert_to_segmentation_mask(mask)

    return obj_id_to_mask #mask


if __name__ == "__main__":
    input_dir = "/home/azureuser/cloudfiles/code/Users/rupaljain/vision_datasets/VOC2007/SegmentationClass/"
    output_dir = "/home/azureuser/cloudfiles/code/Users/rupaljain/vision_datasets/VOC2007/labels/"
    image_dir = "/home/azureuser/cloudfiles/code/Users/rupaljain/vision_datasets/VOC2007/JPEGImages/"

    # create the labels folder (output directory)
    os.makedirs(output_dir, exist_ok=True)

    # identify all the xml files in the annotations folder (input directory)
    png_files = glob.glob(os.path.join(input_dir, '*.png'))
    # loop through each 
    for index, png_file in enumerate(png_files):
        basename = os.path.basename(png_file)
        filename = os.path.splitext(basename)[0]
        # check if the label contains the corresponding image file
        if not os.path.exists(os.path.join(image_dir, f"{filename}.jpg")):
            print(f"{filename} image does not exist!")
            continue

        if index % 1000 == 0:
            print("Processing label", index)

        obj_id_to_mask = binarise_mask(png_file)
        output_list = []
        for class_name, class_mask in obj_id_to_mask.items():
            y_indices, x_indices = np.where(class_mask != 0)

            h, w = class_mask.shape
            normalised_y_indices = [y_index/h for y_index in y_indices]
            normalised_x_indices = [x_index/w for x_index in x_indices]
            curr_mask_line = [str(class_name)]
            for x_index, y_index in zip(normalised_x_indices, normalised_y_indices):
                curr_mask_line.extend([str(x_index), str(y_index)])
            
            curr_mask_line = " ".join(curr_mask_line)
            output_list.append(curr_mask_line)

        if output_list:
            # generate a YOLO format text file for each polygon file
            save_path = os.path.join(output_dir, f"{filename}.txt")
            # print(save_path)
            with open(save_path, "w", encoding="utf-8") as f:
                f.write("\n".join(output_list))