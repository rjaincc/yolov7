from email.mime import image
import glob
import os
import json

CLASSES_OBJECTS = [
    "dog", "person", "train", "sofa", "chair", "car", "pottedplant", "diningtable", "horse", "cat", "cow", "bus", 
    "bicycle", "aeroplane", "motorbike", "tvmonitor", "bird", "bottle", "boat", "sheep"
]


def process_jsonl(jsonl_path: str, output_path: str):
    with open(jsonl_path, "r") as json_file:
        json_list = list(json_file)

    for json_str in json_list:

        out_result = []

        result = json.loads(json_str)
        # print(result)

        image_name = result["image_url"].split("/")[-1]
        label_file_name = image_name.split(".")[0] + ".txt"

        curr_labels = result["label"]

        for curr_label in curr_labels:
            curr_result = []

            curr_class = curr_label["label"]
            curr_class_int = CLASSES_OBJECTS.index(curr_class)

            curr_polygons = curr_label["polygon"]
            curr_polygon = []
            # Not the best thing to do - for quick start we just merge all polygons for now
            for poly in curr_polygons:
                curr_polygon.extend(poly)

            curr_result.append(curr_class_int)
            curr_result.extend(curr_polygon)
            curr_result = [str(i) for i in curr_result]
            # print(curr_class_int, len(curr_polygon), len(curr_result))
            curr_result = " ".join(curr_result)
            out_result.append(curr_result)

        if out_result:
            # generate a YOLO format text file for each polygon file
            save_path = os.path.join(output_dir, f"{label_file_name}")
            print(save_path)
            with open(save_path, "w", encoding="utf-8") as f:
                f.write("\n".join(out_result))


if __name__ == "__main__":
    val_jsonl_path = "/home/azureuser/cloudfiles/code/Users/rupaljain/vision_datasets/VOC2007/voc2007_val_polygon.jsonl"
    train_jsonl_path = "/home/azureuser/cloudfiles/code/Users/rupaljain/vision_datasets/VOC2007/voc2007_train_polygon.jsonl"

    # We don't have test json-l for VOC 
    test_jsonl_path = "/home/azureuser/cloudfiles/code/Users/rupaljain/vision_datasets/VOC2007/voc2007_test_polygon.jsonl"

    output_dir = "/home/azureuser/cloudfiles/code/Users/rupaljain/vision_datasets/VOC2007/labels/"

    # create the labels folder (output directory)
    os.makedirs(output_dir, exist_ok=True)


    # process_jsonl(val_jsonl_path, output_path=output_dir)

    # process_jsonl(train_jsonl_path, output_path=output_dir)

    # process_jsonl(test_jsonl_path, output_path=output_dir)