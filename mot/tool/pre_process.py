import numpy as np
import pickle
import json
import os
from utils import set_dir

src_path = "../../datasets/algorithm_results/detect_merge/"

cam_names = ['c041','c042','c043','c044','c045','c046']
# cam_names = ['c041']

def save_file(data,cam_name,image_name):
    save_path = os.path.join(src_path,cam_name)
    save_path = os.path.join(save_path,"feature")
    save_path = os.path.join(save_path,image_name+".json")
    with open(save_path, 'w') as f:
        json.dump(data, f, cls=NumpyEncoder)
        
class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
                              np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

for cam_name in cam_names:
    cache_file = os.path.join(src_path,cam_name)
    set_dir(os.path.join(cache_file,"feature"))
    set_dir(os.path.join(cache_file,"detect_result"))
    set_dir(os.path.join(cache_file,"mid_result"))
    set_dir(os.path.join(cache_file,"traj_result"))
    set_dir(os.path.join(cache_file,"track_result"))

    cache_file = os.path.join(cache_file,cam_name + "_dets_feat.pkl")
    with open(cache_file, 'rb') as fid:
        data = pickle.load(fid)
        frame_data = {}
        last_image_name = ""
        for index,key in enumerate(data):
            image_name = key.split("_")[0]
            if index == 0:
                frame_data[key] = data[key]
                last_image_name = image_name
                continue
            if last_image_name != image_name:
                save_file(frame_data,cam_name,last_image_name)
                frame_data = {}
                last_image_name = image_name
            frame_data[key] = data[key]
            if index == len(data) -1:
                save_file(frame_data,cam_name,last_image_name)

            
            