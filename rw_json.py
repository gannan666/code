import json
import os
from tqdm import tqdm
import glob


def get_score(obj):
    score = obj["score"]
    return score


def del_obj(json_path):
    del_obj = []
    json_data = json.loads(open(json_path).read())
    objects = json_data["objects"]
    for obj in objects:
        if obj["pointCount"] < 30:
            del_obj.append(obj)
    for x in del_obj:
        objects.remove(x)
    with open("json_dir/{}".format(os.path.basename(json_path)), 'w') as f:
        json.dump(json_data, f)
    

if __name__ == "__main__":
    json_paths = glob.glob("json_dir/*.json")
    json_paths = sorted(json_paths)
    for json_path in json_paths:
        print(json_path)
        del_obj(json_path)