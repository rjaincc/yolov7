import os
import json 

def json_decode_many(s):
  import json
  import json.decoder
  decoder = json.JSONDecoder()
  _w = json.decoder.WHITESPACE.match

  idx = 0

  while True:
    idx = _w(s, idx).end() # skip leading whitespace
    if idx >= len(s):
      break
    obj, idx = decoder.raw_decode(s, idx=idx)
    yield obj
    

def generate_text_file(json_file_path: str):
    data_dir_path = os.path.dirname(json_file_path)
    data_lines = [json.loads(line) for line in open(json_file_path, "r", encoding="utf-8")]
    try:
        file_names = [data_line["imageUrl"].split("/")[-1] for data_line in data_lines]
        file_extension = ".json"
    except KeyError:
        # In case of jsonl files 
        file_names = [data_line["image_url"].split("/")[-1] for data_line in data_lines]
        file_extension = ".jsonl"
    print("\tTotal bboxes: ", len(file_names))
    file_names = list(set(file_names))
    print("\Total Images: ", len(file_names))
    with open(json_file_path.replace(file_extension, ".txt"), "w") as f:
        for file_name in file_names:
            f.write(f"{data_dir_path}/images/{file_name}\n")

if __name__ == "__main__":
    val_json_path = "/home/azureuser/cloudfiles/code/Users/rupaljain/vision_datasets/VOC2007/voc2007_val.json"
    train_json_path = "/home/azureuser/cloudfiles/code/Users/rupaljain/vision_datasets/VOC2007/voc2007_train.json"
    test_jsonl_path = "/home/azureuser/cloudfiles/code/Users/rupaljain/vision_datasets/VOC2007/voc2007_test.jsonl"

    print("Val:")
    generate_text_file(json_file_path=val_json_path)

    print("\n\nTrain")
    generate_text_file(json_file_path=train_json_path)

    print("\n\nTest")
    generate_text_file(json_file_path=test_jsonl_path)