from email.mime import image
import glob
import os
import json

CLASSES_OBJECTS = [
    "dog", "person", "train", "sofa", "chair", "car", "pottedplant", "diningtable", "horse", "cat", "cow", "bus", 
    "bicycle", "aeroplane", "motorbike", "tvmonitor", "bird", "bottle", "boat", "sheep"
]


def xml_to_yolo_bbox(bbox, w, h):
    # xmin, ymin, xmax, ymax
    x_center = ((bbox[2] + bbox[0]) / 2) / w
    y_center = ((bbox[3] + bbox[1]) / 2) / h
    width = (bbox[2] - bbox[0]) / w
    height = (bbox[3] - bbox[1]) / h
    return [x_center, y_center, width, height]


def process_jsonl(jsonl_path: str, output_path: str):
    with open(jsonl_path, "r") as json_file:
        json_list = list(json_file)

    for json_str in json_list[:1]:

        out_result = []

        result = json.loads(json_str)
        # print(result)

        image_name = result["image_url"].split("/")[-1]
        label_file_name = image_name.split(".")[0] + ".txt"

        image_details = result["image_details"]
        image_w = int(image_details["width"].replace("px", ""))
        image_h = int(image_details["height"].replace("px", ""))

        for curr_label in result["label"]:
            curr_result = []

            curr_class = curr_label["label"]
            curr_class_int = CLASSES_OBJECTS.index(curr_class)

            bbox = [
               curr_label["topX"], curr_label["topY"], curr_label["bottomX"], curr_label["bottomY"]
            ]
            # Already normalised bbox coordinates in jsonl. Set w=1, h=1
            yolo_bbox = xml_to_yolo_bbox(bbox, w=1, h=1) 

            curr_result.append(curr_class_int)
            curr_result.extend(yolo_bbox)
            curr_result = [str(i) for i in curr_result]
            curr_result = " ".join(curr_result)

            out_result.append(curr_result)

        if out_result:
            # generate a YOLO format text file for each polygon file
            save_path = os.path.join(output_dir, f"{label_file_name}")
            print(save_path)
            with open(save_path, "w", encoding="utf-8") as f:
                f.write("\n".join(out_result))


if __name__ == "__main__":
    val_jsonl_path = "/home/azureuser/cloudfiles/code/Users/rupaljain/vision_datasets/VOC2007/voc2007_val.jsonl"
    train_jsonl_path = "/home/azureuser/cloudfiles/code/Users/rupaljain/vision_datasets/VOC2007/voc2007_train.jsonl"

    # We don't have test json-l for VOC 
    test_jsonl_path = "/home/azureuser/cloudfiles/code/Users/rupaljain/vision_datasets/VOC2007/voc2007_test.jsonl"

    output_dir = "/home/azureuser/cloudfiles/code/Users/rupaljain/vision_datasets/VOC2007/labels/"

    # create the labels folder (output directory)
    os.makedirs(output_dir, exist_ok=True)


    process_jsonl(val_jsonl_path, output_path=output_dir)

    process_jsonl(train_jsonl_path, output_path=output_dir)

    process_jsonl(test_jsonl_path, output_path=output_dir)