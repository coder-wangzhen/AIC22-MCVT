import os
import sys
from utils import set_dir
sys.path.append("..")

from tool.utils import load_mask,load_tracklet,move_tracklet,load_reconn_region,move_keep_tracklet,move_time_tracklet
from tool.filter_tracklet import filter_tracklet_distance,filter_tracklet_region,keep_tracklet_region,remove_tracklet_region,check_tracklet_region,modify_tracklet_region
from tool.joint_tracklets import cam_joint_tracklets,copy_joint_tracklet

base_path = "../../datasets/algorithm_results/detect_merge/"
base_reconn_region = "../re_region"
if __name__=='__main__':
    cam_name_list = ["c041","c042","c043","c044","c045","c046"]
    new_all_mask = load_mask("../mask")
    reconn_regions = load_reconn_region(base_reconn_region)
    for cam_name in cam_name_list:

        set_dir(os.path.join(base_path,cam_name,"connect"))
        set_dir(os.path.join(base_path,cam_name,"connect_image"))
        set_dir(os.path.join(base_path,cam_name,"remove"))
        set_dir(os.path.join(base_path,cam_name,"result"))
        copy_joint_tracklet(cam_name,base_path,"/traj_result/","/result/")

        track_path = os.path.join(base_path,cam_name+"/result/")
        cam_name_tracklets = {}

        print("start to load********************************")
        tracklets = load_tracklet(track_path)
        cam_name_tracklets[cam_name] = tracklets
        print("end to load********************************")

        if cam_name == "c044":
            continue

        print("start to modify_tracklet_region********************************")
        modify_tracklet_region(cam_name_tracklets)
        print("end to modify_tracklet_region********************************")

        print("start to check_tracklet_region********************************")
        check_tracklet_region(cam_name_tracklets)  #异常轨迹过滤
        print("end to check_tracklet_region********************************\r\n")

        print("start to filter_tracklet_distance********************************")
        tracklets = load_tracklet(track_path)
        cam_name_tracklets[cam_name] = tracklets
        remove_tracklets = filter_tracklet_distance(cam_name_tracklets,new_all_mask)  #异常轨迹过滤
        move_tracklet(remove_tracklets,cam_name,base_path,"/remove/")
        print("end to filter_tracklet_distance********************************\r\n")

        print("start to joint********************************")
        tracklets = load_tracklet(track_path)
        cam_name_tracklets[cam_name] = tracklets
        cam_joint_tracklets(cam_name_tracklets,new_all_mask,reconn_regions,base_path)
        print("end to joint********************************\r\n")

       


    
