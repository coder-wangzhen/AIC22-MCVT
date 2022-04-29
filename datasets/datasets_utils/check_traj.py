import os
import json

src_base_path ="/mnt/LocalDisk1/Projects/AIC21-MTMC/datasets/algorithm_results/sot_result/result_0422"
compare_base_path = "/mnt/LocalDisk1/Projects/AIC21-MTMC/datasets/algorithm_results/detect_merge" 
cam_names = ["c041","c042","c043","c044","c045","c046"]
#cam_names = ["c042"]
for cam_name in cam_names:
    traj_path = os.path.join(src_base_path,cam_name)
    traj_path = os.path.join(traj_path,"traj_result")
    traj_files = os.listdir(traj_path)
    diff= []
    diff_ = []
    for file in traj_files:
        compare_path = os.path.join(compare_base_path,cam_name)
        compare_path = os.path.join(compare_path,"result")
        compare_file = os.path.join(compare_path,file)
        if not os.path.exists(compare_file):
            print("not exist compare_file:",compare_file)
            diff_.append(compare_file)
            #raise
        else:
            src_data = {}
            compare_data = {}
            src_traj_file = os.path.join(traj_path,file)
            with open(src_traj_file, 'r') as f:
                src_data = json.load(f)
            with open(compare_file, 'r') as compare_f:
                compare_data = json.load(compare_f)
            if (src_data["feature_list"] != compare_data["feature_list"]):
                print("compare_file:",compare_file)
                diff.append(compare_file)
                #raise
    print("cam_name:{},diff file:{}".format(cam_name,diff))
    print("cam_name:{},not exist diff file:{}".format(cam_name,diff_))
print("done")
            