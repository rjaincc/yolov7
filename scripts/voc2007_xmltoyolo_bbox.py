import xml.etree.ElementTree as ET
import glob
import os
import json


def xml_to_yolo_bbox(bbox, w, h):
    # xmin, ymin, xmax, ymax
    x_center = ((bbox[2] + bbox[0]) / 2) / w
    y_center = ((bbox[3] + bbox[1]) / 2) / h
    width = (bbox[2] - bbox[0]) / w
    height = (bbox[3] - bbox[1]) / h
    return [x_center, y_center, width, height]


def yolo_to_xml_bbox(bbox, w, h):
    # x_center, y_center width height
    w_half_len = (bbox[2] * w) / 2
    h_half_len = (bbox[3] * h) / 2
    xmin = int((bbox[0] * w) - w_half_len)
    ymin = int((bbox[1] * h) - h_half_len)
    xmax = int((bbox[0] * w) + w_half_len)
    ymax = int((bbox[1] * h) + h_half_len)
    return [xmin, ymin, xmax, ymax]


if __name__ == "__main__":
    """
    Prerequisite - Download the dataset on the VM first. 
    Todo - Change the locations accordingly. 
    """


    classes = []
    input_dir = "/home/azureuser/cloudfiles/code/Users/rupaljain/vision_datasets/VOC2007/Annotations/"
    output_dir = "/home/azureuser/cloudfiles/code/Users/rupaljain/vision_datasets/VOC2007/labels/"
    image_dir = "/home/azureuser/cloudfiles/code/Users/rupaljain/vision_datasets/VOC2007/images/"

    # create the labels folder (output directory)
    os.makedirs(output_dir, exist_ok=True)

    # identify all the xml files in the annotations folder (input directory)
    xml_files = glob.glob(os.path.join(input_dir, '*.xml'))
    # loop through each 
    for index, xml_file in enumerate(xml_files):
        
        basename = os.path.basename(xml_file)
        filename = os.path.splitext(basename)[0]
        # check if the label contains the corresponding image file
        if not os.path.exists(os.path.join(image_dir, f"{filename}.jpg")):
            print(f"{filename} image does not exist!")
            continue

        if index % 1000 == 0:
            print("Processing label", index)
        result = []

        # parse the content of the xml file
        tree = ET.parse(xml_file)
        root = tree.getroot()
        width = int(root.find("size").find("width").text)
        height = int(root.find("size").find("height").text)

        for obj in root.findall('object'):
            label = obj.find("name").text
            # check for new classes and append to list
            if label not in classes:
                classes.append(label)
            index = classes.index(label)
            pil_bbox = [int(x.text) for x in obj.find("bndbox")]
            yolo_bbox = xml_to_yolo_bbox(pil_bbox, width, height)
            # convert data to string
            bbox_string = " ".join([str(x) for x in yolo_bbox])
            result.append(f"{index} {bbox_string}")
        
            # print(f"{index} {pil_bbox} {width} {height} {bbox_string}")
        if result:
            # generate a YOLO format text file for each xml file
            with open(os.path.join(output_dir, f"{filename}.txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(result))

    # generate the classes file as reference
    with open('classes.txt', 'w', encoding='utf8') as f:
        f.write(json.dumps(classes))