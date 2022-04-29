from ast import Return
from operator import truediv
import os
import json
from pickle import FALSE
import numpy as np
import cv2 
import shutil
import math
from tool.utils import cal_min_iou,is_interaction_out,cal_boxes_avg_area,distance_bet_two

cam_conn_track_id = 1000
total = 2000
cam_dis = {"c041-c042":351,"c042-c043":160,"c043-c044":304,"c044-c045":112,"c045-c046":224}
cam_avg_erae = {"c041":700,"c042":700,"c043":700,"c044":700,"c045":700,"c046":700}


def modify_tracklet_region(all_tracklet):
    end_region_ids = [2,4,6,8]
    for cam_name in all_tracklet:
        if cam_name == "c041":
            for tracklet in all_tracklet[cam_name]:
                start_region_id = tracklet["start_region_id"]
                end_region_id = tracklet["end_region_id"]
                zone_list = tracklet["go_through_region"]
                track_id = tracklet["track_id"]
                end_frame_id = tracklet["end_frame_id"]
                
                if  zone_list[-1] not in end_region_ids and len([i for i in zone_list if i in end_region_ids]) > 0:
                    if end_frame_id != 2000:
                        for region_id in reversed(zone_list):
                            if region_id in end_region_ids:
                                zone_list[-1] = region_id
                                tracklet["end_region_id"] = region_id
                                print("cam:{} warning:{}".format(cam_name,track_id))

        if cam_name == "c042":
            for tracklet in all_tracklet[cam_name]:
                start_region_id = tracklet["start_region_id"]
                end_region_id = tracklet["end_region_id"]
                zone_list = tracklet["go_through_region"]
                track_id = tracklet["track_id"]
                end_frame_id = tracklet["end_frame_id"]
                if  zone_list[-1] not in end_region_ids and len([i for i in zone_list if i in end_region_ids]) > 0:
                    if end_frame_id != 2000:
                        for region_id in reversed(zone_list):
                            if region_id in end_region_ids:
                                zone_list[-1] = region_id
                                tracklet["end_region_id"] = region_id
                                print("cam:{} warning:{}".format(cam_name,track_id))
        if cam_name == "c043":
            for tracklet in all_tracklet[cam_name]:
                start_region_id = tracklet["start_region_id"]
                end_region_id = tracklet["end_region_id"]
                zone_list = tracklet["go_through_region"]
                track_id = tracklet["track_id"]
                end_frame_id = tracklet["end_frame_id"]
                if  zone_list[-1] not in end_region_ids and len([i for i in zone_list if i in end_region_ids]) > 0:
                    if end_frame_id != 2000:
                        for region_id in reversed(zone_list):
                            if region_id in end_region_ids:
                                zone_list[-1] = region_id
                                tracklet["end_region_id"] = region_id
                                print("cam:{} warning:{}".format(cam_name,track_id))
        if cam_name == "c044":
            for tracklet in all_tracklet[cam_name]:
                start_region_id = tracklet["start_region_id"]
                end_region_id = tracklet["end_region_id"]
                zone_list = tracklet["go_through_region"]
                track_id = tracklet["track_id"]
                end_frame_id = tracklet["end_frame_id"]
                if  zone_list[-1] not in end_region_ids and len([i for i in zone_list if i in end_region_ids]) > 0:
                    if end_frame_id != 2000:
                        for region_id in reversed(zone_list):
                            if region_id in end_region_ids:
                                zone_list[-1] = region_id
                                tracklet["end_region_id"] = region_id
                                print("cam:{} warning:{}".format(cam_name,track_id))
        if cam_name == "c045":
            for tracklet in all_tracklet[cam_name]:
                start_region_id = tracklet["start_region_id"]
                end_region_id = tracklet["end_region_id"]
                zone_list = tracklet["go_through_region"]
                track_id = tracklet["track_id"]
                end_frame_id = tracklet["end_frame_id"]
                
                if  zone_list[-1] not in end_region_ids and len([i for i in zone_list if i in end_region_ids]) > 0:
                    if end_frame_id != 2000:
                        for region_id in reversed(zone_list):
                            if region_id in end_region_ids:
                                zone_list[-1] = region_id
                                tracklet["end_region_id"] = region_id
                                print("cam:{} warning:{}".format(cam_name,track_id))
        if cam_name == "c046":
            for tracklet in all_tracklet[cam_name]:
                start_region_id = tracklet["start_region_id"]
                end_region_id = tracklet["end_region_id"]
                zone_list = tracklet["go_through_region"]
                track_id = tracklet["track_id"]
                end_frame_id = tracklet["end_frame_id"]
                if  zone_list[-1] not in end_region_ids and len([i for i in zone_list if i in end_region_ids]) > 0:
                    if end_frame_id != 2000:
                        for region_id in reversed(zone_list):
                            if region_id in region_id:
                                zone_list[-1] = end_region_ids
                                tracklet["end_region_id"] = region_id
                                print("cam:{} warning:{}".format(cam_name,track_id))

def check_tracklet_region(all_tracklet):
    end_region_ids = [2,4,6,8]
    for cam_name in all_tracklet:
        if cam_name == "c041":
            for tracklet in all_tracklet[cam_name]:
                start_region_id = tracklet["start_region_id"]
                end_region_id = tracklet["end_region_id"]
                zone_list = tracklet["go_through_region"]
                track_id = tracklet["track_id"]
                end_frame_id = tracklet["end_frame_id"]
                if len(zone_list) == 0:
                    print("cam:{},go_through_region is error track id:{}".format(cam_name,track_id))
                    raise
                # if len(zone_list) >= 4:
                #     print("cam:{},go_through_region size:{}, track id:{}".format(cam_name,len(zone_list),track_id))
                #     raise
                if zone_list[-1] != end_region_id:
                    print("cam:{},zone_list[-1] != end_region_id go_through_region ERROR:{}, track id:{}".format(cam_name,zone_list,track_id))
                    raise
                if zone_list[0] != start_region_id:
                    print("cam:{},zone_list[0] != start_region_id go_through_region ERROR:{}, track id:{}".format(cam_name,zone_list,track_id))
                    raise
                if  zone_list[-1] not in end_region_ids and len([i for i in zone_list if i in end_region_ids]) > 0:
                    if end_frame_id != 2000:
                        print("cam:{},go_through_region ERROR:{}, track id:{}".format(cam_name,zone_list,track_id))
                        raise
        if cam_name == "c042":
            for tracklet in all_tracklet[cam_name]:
                start_region_id = tracklet["start_region_id"]
                end_region_id = tracklet["end_region_id"]
                zone_list = tracklet["go_through_region"]
                track_id = tracklet["track_id"]
                end_frame_id = tracklet["end_frame_id"]
                if len(zone_list) == 0:
                    print("cam:{},go_through_region is error track id:{}".format(cam_name,track_id))
                    raise
                # if len(zone_list) >= 4:
                #     print("cam:{},go_through_region size:{}, track id:{}".format(cam_name,len(zone_list),track_id))
                #     raise
                if zone_list[-1] != end_region_id:
                    print("cam:{},zone_list[-1] != end_region_id go_through_region ERROR:{}, track id:{}".format(cam_name,zone_list,track_id))
                    raise
                if zone_list[0] != start_region_id:
                    print("cam:{},zone_list[0] != start_region_id go_through_region ERROR:{}, track id:{}".format(cam_name,zone_list,track_id))
                    raise
                if  zone_list[-1] not in end_region_ids and len([i for i in zone_list if i in end_region_ids]) > 0:
                    if end_frame_id != 2000:
                        print("cam:{},go_through_region ERROR:{}, track id:{}".format(cam_name,zone_list,track_id))
                        raise
        if cam_name == "c043":
            for tracklet in all_tracklet[cam_name]:
                start_region_id = tracklet["start_region_id"]
                end_region_id = tracklet["end_region_id"]
                zone_list = tracklet["go_through_region"]
                track_id = tracklet["track_id"]
                end_frame_id = tracklet["end_frame_id"]
                if len(zone_list) == 0:
                    print("cam:{},go_through_region is error track id:{}".format(cam_name,track_id))
                    raise
                # if len(zone_list) >= 4:
                #     print("cam:{},go_through_region size:{}, track id:{}".format(cam_name,len(zone_list),track_id))
                #     raise
                if zone_list[-1] != end_region_id:
                    print("cam:{},zone_list[-1] != end_region_id go_through_region ERROR:{}, track id:{}".format(cam_name,zone_list,track_id))
                    raise
                if zone_list[0] != start_region_id:
                    print("cam:{},zone_list[0] != start_region_id go_through_region ERROR:{}, track id:{}".format(cam_name,zone_list,track_id))
                    raise
                if  zone_list[-1] not in end_region_ids and len([i for i in zone_list if i in end_region_ids]) > 0:
                    if end_frame_id != 2000:
                        print("cam:{},go_through_region ERROR:{}, track id:{}".format(cam_name,zone_list,track_id))
                        raise
        if cam_name == "c044":
            for tracklet in all_tracklet[cam_name]:
                start_region_id = tracklet["start_region_id"]
                end_region_id = tracklet["end_region_id"]
                zone_list = tracklet["go_through_region"]
                track_id = tracklet["track_id"]
                end_frame_id = tracklet["end_frame_id"]
                if len(zone_list) == 0:
                    print("cam:{},go_through_region is error track id:{}".format(cam_name,track_id))
                    raise
                # if len(zone_list) >= 4:
                #     print("cam:{},go_through_region size:{}, track id:{}".format(cam_name,len(zone_list),track_id))
                #     raise
                if zone_list[-1] != end_region_id:
                    print("cam:{},zone_list[-1] != end_region_id go_through_region ERROR:{}, track id:{}".format(cam_name,zone_list,track_id))
                    raise
                if zone_list[0] != start_region_id:
                    print("cam:{},zone_list[0] != start_region_id go_through_region ERROR:{}, track id:{}".format(cam_name,zone_list,track_id))
                    raise
                if  zone_list[-1] not in end_region_ids and len([i for i in zone_list if i in end_region_ids]) > 0:
                    if end_frame_id != 2000:
                        print("cam:{},go_through_region ERROR:{}, track id:{}".format(cam_name,zone_list,track_id))
                        raise
        if cam_name == "c045":
            for tracklet in all_tracklet[cam_name]:
                start_region_id = tracklet["start_region_id"]
                end_region_id = tracklet["end_region_id"]
                zone_list = tracklet["go_through_region"]
                track_id = tracklet["track_id"]
                end_frame_id = tracklet["end_frame_id"]
                
                if len(zone_list) == 0:
                    print("cam:{},go_through_region is error track id:{}".format(cam_name,track_id))
                    raise
                # if len(zone_list) >= 4:
                #     print("cam:{},go_through_region size:{}, track id:{}".format(cam_name,len(zone_list),track_id))
                #     raise
                if zone_list[-1] != end_region_id:
                    print("cam:{},zone_list[-1] != end_region_id go_through_region ERROR:{}, track id:{}".format(cam_name,zone_list,track_id))
                    raise
                if zone_list[0] != start_region_id:
                    print("cam:{},zone_list[0] != start_region_id go_through_region ERROR:{}, track id:{}".format(cam_name,zone_list,track_id))
                    raise
                if  zone_list[-1] not in end_region_ids and len([i for i in zone_list if i in end_region_ids]) > 0:
                    if end_frame_id != 2000:
                        print("cam:{},go_through_region ERROR:{}, track id:{}".format(cam_name,zone_list,track_id))
                        raise
        if cam_name == "c046":
            for tracklet in all_tracklet[cam_name]:
                start_region_id = tracklet["start_region_id"]
                end_region_id = tracklet["end_region_id"]
                zone_list = tracklet["go_through_region"]
                track_id = tracklet["track_id"]
                end_frame_id = tracklet["end_frame_id"]
                if track_id == 10:
                    continue
                if len(zone_list) == 0:
                    print("cam:{},go_through_region is error track id:{}".format(cam_name,track_id))
                    raise
                # if len(zone_list) >= 4:
                #     print("cam:{},go_through_region size:{}, track id:{}".format(cam_name,len(zone_list),track_id))
                #     raise
                if zone_list[-1] != end_region_id:
                    print("cam:{},zone_list[-1] != end_region_id go_through_region ERROR:{}, track id:{}".format(cam_name,zone_list,track_id))
                    raise
                if zone_list[0] != start_region_id:
                    print("cam:{},zone_list[0] != start_region_id go_through_region ERROR:{}, track id:{}".format(cam_name,zone_list,track_id))
                    raise
                if  zone_list[-1] not in end_region_ids and len([i for i in zone_list if i in end_region_ids]) > 0:
                    if end_frame_id != 2000:
                        print("cam:{},go_through_region ERROR:{}, track id:{}".format(cam_name,zone_list,track_id))
                        raise


def filter_tracklet_region(all_tracklet):
    for cam_name in all_tracklet:
        if cam_name == "c041":
            cam_41_remove_tracklet = {"region":[],"same_region_p":[],"same_region_l":[]}
            for tracklet in all_tracklet[cam_name]:
                start_region_id = tracklet["start_region_id"]
                end_region_id = tracklet["end_region_id"]
                start_frame_id = tracklet["start_frame_id"]
                end_frame_id = tracklet["end_frame_id"]
                if start_region_id == end_region_id:
                    if start_region_id in [3,4,7,8]:
                        cam_41_remove_tracklet["same_region_p"].append(tracklet["track_id"])
                        continue
                    if end_frame_id - start_frame_id:
                        cam_41_remove_tracklet["same_region_l"].append(tracklet["track_id"])
                        continue
                if(start_region_id == 3) and (end_region_id == 8):
                    cam_41_remove_tracklet["region"].append(tracklet["track_id"])
                    continue
                if(start_region_id == 5) and (end_region_id == 4):
                    cam_41_remove_tracklet["region"].append(tracklet["track_id"])
                    continue
                if(start_region_id == 3) and (end_region_id == 6):
                    cam_41_remove_tracklet["region"].append(tracklet["track_id"])
                    continue
                if(start_region_id == 5) and (end_region_id == 8):
                    cam_41_remove_tracklet["region"].append(tracklet["track_id"])
                    continue
                if(start_region_id == 7) and (end_region_id == 4):
                    cam_41_remove_tracklet["region"].append(tracklet["track_id"])
                    continue
                if(start_region_id == 7) and (end_region_id == 6):
                    cam_41_remove_tracklet["region"].append(tracklet["track_id"])
                    continue
            return  cam_41_remove_tracklet
        if cam_name == "c042":
            cam_42_remove_tracklet = {"region":[],"same_region_p":[],"same_region_l":[]}
            for tracklet in all_tracklet[cam_name]:
                start_region_id = tracklet["start_region_id"]
                end_region_id = tracklet["end_region_id"]
                start_frame_id = tracklet["start_frame_id"]
                end_frame_id = tracklet["end_frame_id"]
                if start_region_id == end_region_id:
                    if start_region_id in [3,4,7,8]:
                        cam_42_remove_tracklet["same_region_p"].append(tracklet["track_id"])
                        continue
                    if end_frame_id - start_frame_id:
                        cam_42_remove_tracklet["same_region_l"].append(tracklet["track_id"])
                        continue
                if(start_region_id == 3) and (end_region_id == 8):
                    cam_42_remove_tracklet["region"].append(tracklet["track_id"])
                    continue
                if(start_region_id == 7) and (end_region_id == 4):
                    cam_42_remove_tracklet["region"].append(tracklet["track_id"])
                    continue
            return  cam_42_remove_tracklet
        if cam_name == "c043":
            cam_43_remove_tracklet = {"region":[],"same_region_p":[],"same_region_l":[]}
            for tracklet in all_tracklet[cam_name]:
                start_region_id = tracklet["start_region_id"]
                end_region_id = tracklet["end_region_id"]
                start_frame_id = tracklet["start_frame_id"]
                end_frame_id = tracklet["end_frame_id"]
                if start_region_id == end_region_id:
                    if start_region_id in [3,4,7,8]:
                        cam_43_remove_tracklet["same_region_p"].append(tracklet["track_id"])
                        continue
                    if end_frame_id - start_frame_id:
                        cam_43_remove_tracklet["same_region_l"].append(tracklet["track_id"])
                        continue
                if(start_region_id == 3) and (end_region_id == 8):
                    cam_43_remove_tracklet["region"].append(tracklet["track_id"])
                    continue
                if(start_region_id == 7) and (end_region_id == 4):
                    cam_43_remove_tracklet["region"].append(tracklet["track_id"])
                    continue
            return  cam_43_remove_tracklet
        if cam_name == "c044":
            cam_44_remove_tracklet = {"region":[],"same_region_p":[],"same_region_l":[]}
            for tracklet in all_tracklet[cam_name]:
                start_region_id = tracklet["start_region_id"]
                end_region_id = tracklet["end_region_id"]
                start_frame_id = tracklet["start_frame_id"]
                end_frame_id = tracklet["end_frame_id"]
                if start_region_id == end_region_id:
                    if start_region_id in [3,4,7,8]:
                        cam_44_remove_tracklet["same_region_p"].append(tracklet["track_id"])
                        continue
                    if end_frame_id - start_frame_id:
                        cam_44_remove_tracklet["same_region_l"].append(tracklet["track_id"])
                        continue
                if(start_region_id == 1) and (end_region_id == 6):
                    cam_44_remove_tracklet["region"].append(tracklet["track_id"])
                    continue
                if(start_region_id == 5) and (end_region_id == 2):
                    cam_44_remove_tracklet["region"].append(tracklet["track_id"])
                    continue
            return  cam_44_remove_tracklet
        if cam_name == "c045":
            cam_45_remove_tracklet = {"region":[],"same_region_p":[],"same_region_l":[]}
            for tracklet in all_tracklet[cam_name]:
                start_region_id = tracklet["start_region_id"]
                end_region_id = tracklet["end_region_id"]
                start_frame_id = tracklet["start_frame_id"]
                end_frame_id = tracklet["end_frame_id"]
                if start_region_id == end_region_id:
                    if start_region_id in [3,4,7,8]:
                        cam_45_remove_tracklet["same_region_p"].append(tracklet["track_id"])
                        continue
                    if end_frame_id - start_frame_id:
                        cam_45_remove_tracklet["same_region_l"].append(tracklet["track_id"])
                        continue
                if(start_region_id == 1) and (end_region_id == 6):
                    cam_45_remove_tracklet["region"].append(tracklet["track_id"])
                    continue
                if(start_region_id == 5) and (end_region_id == 2):
                    cam_45_remove_tracklet["region"].append(tracklet["track_id"])
                    continue
            return  cam_45_remove_tracklet
        if cam_name == "c046":
            cam_46_remove_tracklet = {"region":[],"same_region_p":[],"same_region_l":[]}
            for tracklet in all_tracklet[cam_name]:
                start_region_id = tracklet["start_region_id"]
                end_region_id = tracklet["end_region_id"]
                start_frame_id = tracklet["start_frame_id"]
                end_frame_id = tracklet["end_frame_id"]
                if start_region_id == end_region_id:
                    if start_region_id in [3,4,7,8]:
                        cam_46_remove_tracklet["same_region_p"].append(tracklet["track_id"])
                        continue
                    if end_frame_id - start_frame_id:
                        cam_46_remove_tracklet["same_region_l"].append(tracklet["track_id"])
                        continue
                if(start_region_id == 1) and (end_region_id == 8):
                    cam_46_remove_tracklet["region"].append(tracklet["track_id"])
                    continue
                if(start_region_id == 1) and (end_region_id == 4):
                    cam_46_remove_tracklet["region"].append(tracklet["track_id"])
                    continue
                if(start_region_id == 3) and (end_region_id == 2):
                    cam_46_remove_tracklet["region"].append(tracklet["track_id"])
                    continue
                if(start_region_id == 3) and (end_region_id == 8):
                    cam_46_remove_tracklet["region"].append(tracklet["track_id"])
                    continue
                if(start_region_id == 7) and (end_region_id == 2):
                    cam_46_remove_tracklet["region"].append(tracklet["track_id"])
                    continue
                if(start_region_id == 7) and (end_region_id == 4):
                    cam_46_remove_tracklet["region"].append(tracklet["track_id"])
                    continue
            return  cam_46_remove_tracklet

def filter_tracklet_distance(all_tracklet,new_all_mask):
    for cam_name in all_tracklet:
        if cam_name == "c041":
            cam_41_remove_tracklet = {"sv":[],"suo":[],"suf":[],"sel":[]}
            for tracklet in all_tracklet[cam_name]:
                is_feature = tracklet["is_feature"]
                boxes = tracklet["box_list"]
                feature_index = np.where(np.array(is_feature) > 0)
                tracklet_valid_i = feature_index[0].tolist()
                tracklet_valid_length = len(tracklet_valid_i)
                start_region_id = tracklet["start_region_id"]
                end_region_id = tracklet["end_region_id"]
                start_frame_id = tracklet["start_frame_id"]
                end_frame_id = tracklet["end_frame_id"]
                if((start_region_id == end_region_id) and (tracklet_valid_length <= 5)):  #有特征的值小于5
                    if end_frame_id != 2000:
                        cam_41_remove_tracklet["sv"].append(tracklet["track_id"])
                        continue
                if(start_region_id == end_region_id):    #框的最小IOU距离大于0.85与外区域相交  去掉场景中没动的车辆 
                    min_iou_distance = cal_min_iou(boxes)
                    if(min_iou_distance >  0.4):
                        if is_interaction_out(boxes,cam_name,new_all_mask):
                            cam_41_remove_tracklet["suo"].append(tracklet["track_id"])
                            continue
                if(start_region_id == end_region_id):    #框的最小IOU距离大于0.85与外区域相交  去掉场景中没动的车辆 
                    min_iou_distance = cal_min_iou(boxes)
                    if(min_iou_distance >  0.4):
                        if end_frame_id - start_frame_id > 1000:
                            cam_41_remove_tracklet["suf"].append(tracklet["track_id"])
                            continue
                if(start_region_id == end_region_id):    #框的最小IOU距离大于0.85与外区域相交  去掉场景中没动的车辆 
                    avg_area = cal_boxes_avg_area(boxes)
                    if(avg_area <=  cam_avg_erae[cam_name]):
                        if(len(boxes) <  50):
                            cam_41_remove_tracklet["sel"].append(tracklet["track_id"])
                            continue
                max_dis = distance_bet_two(boxes)
                if(max_dis > 200):
                    print("****************************cam name:",cam_name)
                    print("****************************check track id:",tracklet["track_id"])
                    raise
            return  cam_41_remove_tracklet
        if cam_name == "c042":
            cam_42_remove_tracklet = {"sv":[],"suo":[],"suf":[],"sel":[],"error_region":[]}
            for tracklet in all_tracklet[cam_name]:
                is_feature = tracklet["is_feature"]
                boxes = tracklet["box_list"]
                feature_index = np.where(np.array(is_feature) > 0)
                tracklet_valid_i = feature_index[0].tolist()
                tracklet_valid_length = len(tracklet_valid_i)
                start_region_id = tracklet["start_region_id"]
                end_region_id = tracklet["end_region_id"]
                start_frame_id = tracklet["start_frame_id"]
                end_frame_id = tracklet["end_frame_id"]
                if((start_region_id == end_region_id) and (tracklet_valid_length <= 5)):  #有特征的值小于5
                    if end_frame_id != 2000:
                        cam_42_remove_tracklet["sv"].append(tracklet["track_id"])
                        continue
                if(start_region_id == end_region_id):    #框的最小IOU距离大于0.85与外区域相交  去掉场景中没动的车辆 
                    min_iou_distance = cal_min_iou(boxes)
                    if(min_iou_distance >  0.4):
                        if is_interaction_out(boxes,cam_name,new_all_mask):
                            cam_42_remove_tracklet["suo"].append(tracklet["track_id"])
                            continue
                if(start_region_id == end_region_id):    #框的最小IOU距离大于0.85与外区域相交  去掉场景中没动的车辆 
                    min_iou_distance = cal_min_iou(boxes)
                    if(min_iou_distance >  0.4):
                        if end_frame_id - start_frame_id > 1000:
                            cam_42_remove_tracklet["suf"].append(tracklet["track_id"])
                            continue
                if(start_region_id == end_region_id):    #框的最小IOU距离大于0.85与外区域相交  去掉场景中没动的车辆 
                    avg_area = cal_boxes_avg_area(boxes)
                    if(avg_area <=  cam_avg_erae[cam_name]):
                        if(len(boxes) <  50):
                            cam_42_remove_tracklet["sel"].append(tracklet["track_id"])
                            continue
                if (start_region_id == 1) and (end_region_id == 2):
                    cam_42_remove_tracklet["error_region"].append(tracklet["track_id"])
                    continue
                max_dis = distance_bet_two(boxes)
                if(max_dis > 200):
                    print("****************************cam name:",cam_name)
                    print("****************************check track id:",tracklet["track_id"])
                    raise
            return  cam_42_remove_tracklet
        if cam_name == "c043":
            cam_43_remove_tracklet = {"sv":[],"suo":[],"suf":[],"sel":[]}
            for tracklet in all_tracklet[cam_name]:
                is_feature = tracklet["is_feature"]
                boxes = tracklet["box_list"]
                feature_index = np.where(np.array(is_feature) > 0)
                tracklet_valid_i = feature_index[0].tolist()
                tracklet_valid_length = len(tracklet_valid_i)
                start_region_id = tracklet["start_region_id"]
                end_region_id = tracklet["end_region_id"]
                start_frame_id = tracklet["start_frame_id"]
                end_frame_id = tracklet["end_frame_id"]
                if((start_region_id == end_region_id) and (tracklet_valid_length <= 5)):  #有特征的值小于5
                    if end_frame_id != 2000:
                        cam_43_remove_tracklet["sv"].append(tracklet["track_id"])
                        continue
                if(start_region_id == end_region_id):    #框的最小IOU距离大于0.85与外区域相交  去掉场景中没动的车辆 
                    min_iou_distance = cal_min_iou(boxes)
                    if(min_iou_distance >  0.4):
                        if is_interaction_out(boxes,cam_name,new_all_mask):
                            cam_43_remove_tracklet["suo"].append(tracklet["track_id"])
                            continue
                if(start_region_id == end_region_id):    #框的最小IOU距离大于0.85与外区域相交  去掉场景中没动的车辆 
                    min_iou_distance = cal_min_iou(boxes)
                    if(min_iou_distance >  0.4):
                        if end_frame_id - start_frame_id > 1000:
                            cam_43_remove_tracklet["suf"].append(tracklet["track_id"])
                            continue
                if(start_region_id == end_region_id):    #框的最小IOU距离大于0.85与外区域相交  去掉场景中没动的车辆 
                    avg_area = cal_boxes_avg_area(boxes)
                    if(avg_area <=  cam_avg_erae[cam_name]):
                        if(len(boxes) <  50):
                            cam_43_remove_tracklet["sel"].append(tracklet["track_id"])
                            continue
                max_dis = distance_bet_two(boxes)
                if(max_dis > 200):
                    print("****************************cam name:",cam_name)
                    print("****************************check track id:",tracklet["track_id"])
                    raise
            return  cam_43_remove_tracklet
        if cam_name == "c044":
            cam_44_remove_tracklet = {"sv":[],"suo":[],"suf":[],"sel":[]}
            for tracklet in all_tracklet[cam_name]:
                is_feature = tracklet["is_feature"]
                boxes = tracklet["box_list"]
                feature_index = np.where(np.array(is_feature) > 0)
                tracklet_valid_i = feature_index[0].tolist()
                tracklet_valid_length = len(tracklet_valid_i)
                start_region_id = tracklet["start_region_id"]
                end_region_id = tracklet["end_region_id"]
                start_frame_id = tracklet["start_frame_id"]
                end_frame_id = tracklet["end_frame_id"]
                if((start_region_id == end_region_id) and (tracklet_valid_length <= 6)):  #有特征的值小于5
                    if end_frame_id != 2000:
                        cam_44_remove_tracklet["sv"].append(tracklet["track_id"])
                        continue
                if(start_region_id == end_region_id):    #框的最小IOU距离大于0.85与外区域相交  去掉场景中没动的车辆 
                    min_iou_distance = cal_min_iou(boxes)
                    if(min_iou_distance >  0.4):
                        if is_interaction_out(boxes,cam_name,new_all_mask):
                            cam_44_remove_tracklet["suo"].append(tracklet["track_id"])
                            continue
                if(start_region_id == end_region_id):    #框的最小IOU距离大于0.85与外区域相交  去掉场景中没动的车辆 
                    min_iou_distance = cal_min_iou(boxes)
                    if(min_iou_distance >  0.4):
                        if end_frame_id - start_frame_id > 1000:
                            cam_44_remove_tracklet["suf"].append(tracklet["track_id"])
                            continue
                if(start_region_id == end_region_id):    #框的最小IOU距离大于0.85与外区域相交  去掉场景中没动的车辆 
                    avg_area = cal_boxes_avg_area(boxes)
                    if(avg_area <=  cam_avg_erae[cam_name]):
                        if(len(boxes) <  50):
                            cam_44_remove_tracklet["sel"].append(tracklet["track_id"])
                            continue
                max_dis = distance_bet_two(boxes)
                if(max_dis > 200):
                    print("****************************cam name:",cam_name)
                    print("****************************check track id:",tracklet["track_id"])
                    raise
            return  cam_44_remove_tracklet
        if cam_name == "c045":
            cam_45_remove_tracklet = {"sv":[],"suo":[],"suf":[],"sel":[]}
            for tracklet in all_tracklet[cam_name]:
                is_feature = tracklet["is_feature"]
                boxes = tracklet["box_list"]
                feature_index = np.where(np.array(is_feature) > 0)
                tracklet_valid_i = feature_index[0].tolist()
                tracklet_valid_length = len(tracklet_valid_i)
                start_region_id = tracklet["start_region_id"]
                end_region_id = tracklet["end_region_id"]
                start_frame_id = tracklet["start_frame_id"]
                end_frame_id = tracklet["end_frame_id"]
                if((start_region_id == end_region_id) and (tracklet_valid_length <= 6)):  #有特征的值小于5
                    if end_frame_id != 2000:
                        cam_45_remove_tracklet["sv"].append(tracklet["track_id"])
                        continue
                if(start_region_id == end_region_id):    #框的最小IOU距离大于0.85与外区域相交  去掉场景中没动的车辆 
                    min_iou_distance = cal_min_iou(boxes)
                    if(min_iou_distance >  0.4):
                        if is_interaction_out(boxes,cam_name,new_all_mask):
                            cam_45_remove_tracklet["suo"].append(tracklet["track_id"])
                            continue
                if(start_region_id == end_region_id):    #框的最小IOU距离大于0.85与外区域相交  去掉场景中没动的车辆 
                    min_iou_distance = cal_min_iou(boxes)
                    if(min_iou_distance >  0.4):
                        if end_frame_id - start_frame_id > 500:
                            cam_45_remove_tracklet["suf"].append(tracklet["track_id"])
                            continue
                if(start_region_id == end_region_id):    #框的最小IOU距离大于0.85与外区域相交  去掉场景中没动的车辆 
                    avg_area = cal_boxes_avg_area(boxes)
                    if(avg_area <=  cam_avg_erae[cam_name]):
                        if(len(boxes) <  50):
                            cam_45_remove_tracklet["sel"].append(tracklet["track_id"])
                            continue
                max_dis = distance_bet_two(boxes)
                if(max_dis > 200):
                    print("****************************cam name:",cam_name)
                    print("****************************check track id:",tracklet["track_id"])
                    raise
            return  cam_45_remove_tracklet
        if cam_name == "c046":
            cam_46_remove_tracklet = {"sv":[],"suo":[],"suf":[],"sel":[]}
            for tracklet in all_tracklet[cam_name]:
                is_feature = tracklet["is_feature"]
                boxes = tracklet["box_list"]
                feature_index = np.where(np.array(is_feature) > 0)
                tracklet_valid_i = feature_index[0].tolist()
                tracklet_valid_length = len(tracklet_valid_i)
                start_region_id = tracklet["start_region_id"]
                end_region_id = tracklet["end_region_id"]
                start_frame_id = tracklet["start_frame_id"]
                end_frame_id = tracklet["end_frame_id"]
                if((start_region_id == end_region_id) and (tracklet_valid_length <= 5)):  #有特征的值小于5
                    if end_frame_id != 2000:
                        cam_46_remove_tracklet["sv"].append(tracklet["track_id"])
                        continue
                if(start_region_id == end_region_id):    #框的最小IOU距离大于0.85与外区域相交  去掉场景中没动的车辆 
                    min_iou_distance = cal_min_iou(boxes)
                    if(min_iou_distance >  0.4):
                        if is_interaction_out(boxes,cam_name,new_all_mask):
                            cam_46_remove_tracklet["suo"].append(tracklet["track_id"])
                            continue
                if(start_region_id == end_region_id):    #框的最小IOU距离大于0.85与外区域相交  去掉场景中没动的车辆 
                    min_iou_distance = cal_min_iou(boxes)
                    if(min_iou_distance >  0.4):
                        if end_frame_id - start_frame_id > 1000:
                            cam_46_remove_tracklet["suf"].append(tracklet["track_id"])
                            continue
                if(start_region_id == end_region_id):    #框的最小IOU距离大于0.85与外区域相交  去掉场景中没动的车辆 
                    avg_area = cal_boxes_avg_area(boxes)
                    if(avg_area <=  cam_avg_erae[cam_name]):
                        if(len(boxes) <  50):
                            cam_46_remove_tracklet["sel"].append(tracklet["track_id"])
                            continue
                max_dis = distance_bet_two(boxes)
                if(max_dis > 200):
                    print("****************************cam name:",cam_name)
                    print("****************************check track id:",tracklet["track_id"])
                    raise
            return  cam_46_remove_tracklet

# 进入漏洞  ： 在startframe 创建 又在限制针级联匹配的漏掉了  
# 出去漏洞： 本来可以在帧数离开，但是由于遮挡 没有离开在在离开后 在限制帧数后匹配的 漏掉了
def keep_tracklet_region(all_tracklet):
    new_all_tracklet = []
    for cam_name in all_tracklet:
        if cam_name == "c041":
            for tracklet in all_tracklet[cam_name]:
                start_region_id = tracklet["start_region_id"]
                end_region_id = tracklet["end_region_id"]
                start_frame_id = tracklet["start_frame_id"]
                end_frame_id = tracklet["end_frame_id"]
                if(((end_region_id == 2) or(end_region_id == 10)) and (end_frame_id < (total-cam_dis["c041-c042"]))):  # 去往 42
                    new_all_tracklet.append(tracklet["track_id"])
                    continue
                if(((start_region_id == 1) or (start_region_id == 10))and (start_frame_id > cam_dis["c041-c042"])): #  接收42
                    new_all_tracklet.append(tracklet["track_id"])
                    continue
            return new_all_tracklet
        if cam_name == "c042":
            for tracklet in all_tracklet[cam_name]:
                start_region_id = tracklet["start_region_id"]
                end_region_id = tracklet["end_region_id"]
                start_frame_id = tracklet["start_frame_id"]
                end_frame_id = tracklet["end_frame_id"]
                if(((start_region_id == 5)or (start_region_id == 10)) and (start_frame_id > cam_dis["c041-c042"])):  # 接收41
                    new_all_tracklet.append(tracklet["track_id"])
                    continue
                if(((end_region_id == 2)or (end_region_id == 10)) and (end_frame_id < total-cam_dis["c042-c043"])): #  去往 43
                    new_all_tracklet.append(tracklet["track_id"])
                    continue
                if(((start_region_id == 1)or (start_region_id == 10)) and (start_frame_id > cam_dis["c042-c043"])):  # 接收43
                    new_all_tracklet.append(tracklet["track_id"])
                    continue
                if(((end_region_id == 6)or (end_region_id == 10)) and (end_frame_id < total-cam_dis["c041-c042"])): #  去往 41
                    new_all_tracklet.append(tracklet["track_id"])
                    continue
            return new_all_tracklet
        if cam_name == "c043":
            for tracklet in all_tracklet[cam_name]:
                start_region_id = tracklet["start_region_id"]
                end_region_id = tracklet["end_region_id"]
                start_frame_id = tracklet["start_frame_id"]
                end_frame_id = tracklet["end_frame_id"]
                if(((start_region_id == 5)or (start_region_id == 10)) and (start_frame_id > cam_dis["c042-c043"])):  # 接收42
                    new_all_tracklet.append(tracklet["track_id"])
                    continue
                if(((end_region_id == 2)or (end_region_id == 10)) and (end_frame_id < total-cam_dis["c043-c044"])): #  去往 44
                    new_all_tracklet.append(tracklet["track_id"])
                    continue
                if(((start_region_id == 1)or (start_region_id == 10)) and (start_frame_id > cam_dis["c043-c044"])):  # 接收44
                    new_all_tracklet.append(tracklet["track_id"])
                    continue
                if(((end_region_id == 6)or (end_region_id == 10)) and (end_frame_id < total-cam_dis["c042-c043"])): #  去往 42
                    new_all_tracklet.append(tracklet["track_id"])
                    continue
            return new_all_tracklet
        if cam_name == "c044":
            for tracklet in all_tracklet[cam_name]:
                start_region_id = tracklet["start_region_id"]
                end_region_id = tracklet["end_region_id"]
                start_frame_id = tracklet["start_frame_id"]
                end_frame_id = tracklet["end_frame_id"]
                if(((start_region_id == 3)or (start_region_id == 10)) and (start_frame_id > cam_dis["c043-c044"])):  # 接收43
                    new_all_tracklet.append(tracklet["track_id"])
                    continue
                if(((end_region_id == 8)or (end_region_id == 10)) and (end_frame_id < total-cam_dis["c044-c045"])): #  去往 45
                    new_all_tracklet.append(tracklet["track_id"])
                    continue
                if(((start_region_id == 7)or (start_region_id == 10)) and (start_frame_id > cam_dis["c044-c045"])):  # 接收45
                    new_all_tracklet.append(tracklet["track_id"])
                    continue
                if(((end_region_id == 4)or (end_region_id == 10)) and (end_frame_id < total-cam_dis["c043-c044"])): #  去往 43
                    new_all_tracklet.append(tracklet["track_id"])
                    continue
            return new_all_tracklet
        if cam_name == "c045":
            for tracklet in all_tracklet[cam_name]:
                start_region_id = tracklet["start_region_id"]
                end_region_id = tracklet["end_region_id"]
                start_frame_id = tracklet["start_frame_id"]
                end_frame_id = tracklet["end_frame_id"]
                if(((start_region_id == 3)or (start_region_id == 10)) and (start_frame_id > cam_dis["c044-c045"])):  # 接收44
                    new_all_tracklet.append(tracklet["track_id"])
                    continue
                if(((end_region_id == 8)or (end_region_id == 10)) and (end_frame_id < total-cam_dis["c045-c046"])): #  去往 46
                    new_all_tracklet.append(tracklet["track_id"])
                    continue
                if(((start_region_id == 7)or (start_region_id == 10)) and (start_frame_id > cam_dis["c045-c046"])):  # 接收46
                    new_all_tracklet.append(tracklet["track_id"])
                    continue
                if(((end_region_id == 4)or (end_region_id == 10)) and (end_frame_id < total-cam_dis["c044-c045"])): #  去往 44
                    new_all_tracklet.append(tracklet["track_id"])
                    continue
            return new_all_tracklet
        if cam_name == "c046":
            for tracklet in all_tracklet[cam_name]:
                start_region_id = tracklet["start_region_id"]
                end_region_id = tracklet["end_region_id"]
                start_frame_id = tracklet["start_frame_id"]
                end_frame_id = tracklet["end_frame_id"]
                if(((start_region_id == 5)or (start_region_id == 10)) and (start_frame_id > cam_dis["c045-c046"])):  # 接收45
                    new_all_tracklet.append(tracklet["track_id"])
                    continue
                if(((end_region_id == 6)or (end_region_id == 10)) and (end_frame_id < total-cam_dis["c045-c046"])): #  去往 45
                    new_all_tracklet.append(tracklet["track_id"])
                    continue
            return new_all_tracklet
        

def remove_tracklet_region(all_tracklet):
    new_all_tracklet = []
    for cam_name in all_tracklet:
        if cam_name == "c041":
            for tracklet in all_tracklet[cam_name]:
                start_region_id = tracklet["start_region_id"]
                end_region_id = tracklet["end_region_id"]
                start_frame_id = tracklet["start_frame_id"]
                end_frame_id = tracklet["end_frame_id"]
                if((end_region_id == 2) and (end_frame_id >= (total-cam_dis["c041-c042"]))):  # 去往 42
                    new_all_tracklet.append(tracklet["track_id"])
                    continue
                if((start_region_id == 1) and (start_frame_id <= cam_dis["c041-c042"])): #  接收42
                    new_all_tracklet.append(tracklet["track_id"])
                    continue
            return new_all_tracklet
        if cam_name == "c042":
            for tracklet in all_tracklet[cam_name]:
                start_region_id = tracklet["start_region_id"]
                end_region_id = tracklet["end_region_id"]
                start_frame_id = tracklet["start_frame_id"]
                end_frame_id = tracklet["end_frame_id"]
                if((start_region_id == 5) and (start_frame_id <= cam_dis["c041-c042"])):  # 接收41
                    new_all_tracklet.append(tracklet["track_id"])
                    continue
                if((end_region_id == 2) and (end_frame_id >= total-cam_dis["c042-c043"])): #  去往 43
                    new_all_tracklet.append(tracklet["track_id"])
                    continue
                if((start_region_id == 1) and (start_frame_id <= cam_dis["c042-c043"])):  # 接收43
                    new_all_tracklet.append(tracklet["track_id"])
                    continue
                if((end_region_id == 6) and (end_frame_id >= total-cam_dis["c041-c042"])): #  去往 41
                    new_all_tracklet.append(tracklet["track_id"])
                    continue
            return new_all_tracklet
        if cam_name == "c043":
            for tracklet in all_tracklet[cam_name]:
                start_region_id = tracklet["start_region_id"]
                end_region_id = tracklet["end_region_id"]
                start_frame_id = tracklet["start_frame_id"]
                end_frame_id = tracklet["end_frame_id"]
                if((start_region_id == 5) and (start_frame_id <= cam_dis["c042-c043"])):  # 接收42
                    new_all_tracklet.append(tracklet["track_id"])
                    continue
                if((end_region_id == 2) and (end_frame_id >= total-cam_dis["c043-c044"])): #  去往 44
                    new_all_tracklet.append(tracklet["track_id"])
                    continue
                if((start_region_id == 1) and (start_frame_id <= cam_dis["c043-c044"])):  # 接收44
                    new_all_tracklet.append(tracklet["track_id"])
                    continue
                if((end_region_id == 6) and (end_frame_id >= total-cam_dis["c042-c043"])): #  去往 42
                    new_all_tracklet.append(tracklet["track_id"])
                    continue
            return new_all_tracklet
        if cam_name == "c044":
            for tracklet in all_tracklet[cam_name]:
                start_region_id = tracklet["start_region_id"]
                end_region_id = tracklet["end_region_id"]
                start_frame_id = tracklet["start_frame_id"]
                end_frame_id = tracklet["end_frame_id"]
                if((start_region_id == 3) and (start_frame_id <= cam_dis["c043-c044"])):  # 接收43
                    new_all_tracklet.append(tracklet["track_id"])
                    continue
                if((end_region_id == 8) and (end_frame_id >= total-cam_dis["c044-c045"])): #  去往 45
                    new_all_tracklet.append(tracklet["track_id"])
                    continue
                if((start_region_id == 7) and (start_frame_id <= cam_dis["c044-c045"])):  # 接收45
                    new_all_tracklet.append(tracklet["track_id"])
                    continue
                if((end_region_id == 4) and (end_frame_id >= total-cam_dis["c043-c044"])): #  去往 43
                    new_all_tracklet.append(tracklet["track_id"])
                    continue
            return new_all_tracklet
        if cam_name == "c045":
            for tracklet in all_tracklet[cam_name]:
                start_region_id = tracklet["start_region_id"]
                end_region_id = tracklet["end_region_id"]
                start_frame_id = tracklet["start_frame_id"]
                end_frame_id = tracklet["end_frame_id"]
                if((start_region_id == 3) and (start_frame_id <= cam_dis["c044-c045"])):  # 接收44
                    new_all_tracklet.append(tracklet["track_id"])
                    continue
                if((end_region_id == 8) and (end_frame_id >= total-cam_dis["c045-c046"])): #  去往 46
                    new_all_tracklet.append(tracklet["track_id"])
                    continue
                if((start_region_id == 7) and (start_frame_id <= cam_dis["c045-c046"])):  # 接收46
                    new_all_tracklet.append(tracklet["track_id"])
                    continue
                if((end_region_id == 4) and (end_frame_id >= total-cam_dis["c044-c045"])): #  去往 44
                    new_all_tracklet.append(tracklet["track_id"])
                    continue
            return new_all_tracklet
        if cam_name == "c046":
            for tracklet in all_tracklet[cam_name]:
                start_region_id = tracklet["start_region_id"]
                end_region_id = tracklet["end_region_id"]
                start_frame_id = tracklet["start_frame_id"]
                end_frame_id = tracklet["end_frame_id"]
                if((start_region_id == 5) and (start_frame_id <= cam_dis["c045-c046"])):  # 接收45
                    new_all_tracklet.append(tracklet["track_id"])
                    continue
                if((end_region_id == 6) and (end_frame_id >= total-cam_dis["c045-c046"])): #  去往 45
                    new_all_tracklet.append(tracklet["track_id"])
                    continue
            return new_all_tracklet






