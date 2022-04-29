import json
import os
import numpy as np
import scipy
from scipy.spatial.distance import cdist
from sklearn.cluster import AgglomerativeClustering
from tool.utils import copy_joint_tracklet,mymovefile,cal_boxes_avg_area,is_interaction_middle_region,is_in_reconnect_region,distance_boxes
import copy
from tool.utils import load_reconn_filter_region,is_in_crowd_region,is_in_crowd_region42
def check_time(row,col,tracklets):
    # if (tracklets[col]["start_frame_id"] - tracklets[row]["end_frame_id"]) < 2:
    #     return False
    # if (tracklets[col]["start_frame_id"] - tracklets[row]["end_frame_id"]) > 500:
    #     return False
    return True

def check_tracklets(tracklets):
    end_region_ids = [2,4,6,8]
    tracklets.sort(key=function)
    remove_tracklet_idx = []
    re_track_id_list = []
    keep_tracklet_idx = []
    track_id_list = []
    for idx in range(len(tracklets)):
        if idx == 0:
            if tracklets[idx]["end_region_id"] in end_region_ids:
                keep_tracklet_idx.append(idx)
                for ind in keep_tracklet_idx:
                    track_id_list.append(tracklets[ind]["track_id"])
                for idx_temp in range(len(tracklets)):
                    if idx_temp not in keep_tracklet_idx:
                        re_track_id_list.append(tracklets[idx_temp]["track_id"])
                return track_id_list,re_track_id_list
        idx_ = idx + 1
        if idx_ == len(tracklets):
            for ind in keep_tracklet_idx:
                track_id_list.append(tracklets[ind]["track_id"])
            for idx_temp in range(len(tracklets)):
                if idx_temp not in keep_tracklet_idx:
                    re_track_id_list.append(tracklets[idx_temp]["track_id"])
            return track_id_list,re_track_id_list
        offset = tracklets[idx]["end_frame_id"] - tracklets[idx_]["start_frame_id"]
        if offset > -400 and offset < 52:
            if check_space(idx,idx_,tracklets):
                if idx not in keep_tracklet_idx:
                    keep_tracklet_idx.append(idx)
                if idx_ not in keep_tracklet_idx:
                    keep_tracklet_idx.append(idx_)
                if tracklets[keep_tracklet_idx[-1]]["end_region_id"] in end_region_ids:
                    for ind in keep_tracklet_idx:
                        track_id_list.append(tracklets[ind]["track_id"])
                    for idx_temp in range(len(tracklets)):
                        if idx_temp not in keep_tracklet_idx:
                            re_track_id_list.append(tracklets[idx_temp]["track_id"])
                    return track_id_list,re_track_id_list
            else:
                if idx in keep_tracklet_idx:
                    remove_tracklet_idx.append(idx_)
                else:
                    remove_tracklet_idx.append(idx)
        else:
            if idx in keep_tracklet_idx:
                remove_tracklet_idx.append(idx_)
            else:
                remove_tracklet_idx.append(idx)
    print("error*****************")
    raise
    


def check_space(idx,idx_,tracklets):
   
    tracklet_0  = tracklets[idx]
    boxes_0 = tracklet_0["box_list"]

    tracklet_1  = tracklets[idx_]
    boxes_1 = tracklet_1["box_list"]

    a = distance_boxes(boxes_0[0],boxes_0[-1])
    b = distance_boxes(boxes_0[0],boxes_1[0])
    c = distance_boxes(boxes_0[-1],boxes_1[0])
    # print("a:",a)
    # print("b:",b)
    # print("c:",c)
    if (c > 500):
        return False
    if(b - a) > -30:
        if (boxes_0[0][0] > boxes_0[-1][0] and (boxes_0[-1][0] - boxes_1[0][0]) > -40):
            return True
        if (boxes_0[0][0] < boxes_0[-1][0] and  (boxes_1[0][0] - boxes_0[-1][0]) > -40):
            return True
    return False

def gate_cost_matrix(cost_matrix, tracklets):

    for row, track_0 in enumerate(tracklets):
        for col, track_1 in enumerate(tracklets):
            if row == col:
                cost_matrix[row,col] = 2
                continue
            # if check_time(row,col,tracklets) == False:
               
            #     cost_matrix[row,col] = 2
            #     continue
            # if check_space(row,col,tracklets) == False:
                
            #     cost_matrix[row,col] = 2
            #     continue
    return cost_matrix

def embedding_distance(tracklet_0, metric='cosine'):

    cost_matrix = np.zeros((len(tracklet_0), len(tracklet_0)), dtype=np.float64)
    
    track_features_0 = np.asarray([track["avg_feature"] for track in tracklet_0], dtype=np.float64)

    track_features_1 = np.asarray([track["avg_feature"] for track in tracklet_0], dtype=np.float64)
    
    cost_matrix = np.maximum(0.0, cdist(track_features_0, track_features_1, metric))  # Nomalized features by ceiling to 0

    return cost_matrix

def get_cost_matrix(tracklet_0):
    cost_matrix = embedding_distance(tracklet_0)
    cost_matrix = gate_cost_matrix(cost_matrix,tracklet_0)
    return cost_matrix

def get_match(cluster_labels):
    cluster_dict = dict()
    cluster = list()
    result=[]
    for i, l in enumerate(cluster_labels):
        if l in list(cluster_dict.keys()):
            cluster_dict[l].append(i)
        else:
            cluster_dict[l] = [i]
    for idx in cluster_dict:
        cluster.append(cluster_dict[idx])
    print("cluster:",cluster)
    for connect in cluster:
        if len(connect) > 1:
            result.append(connect)
    return result

def check_tracklet(tracklet):
    is_box_list = tracklet["is_box"]
    box_list = tracklet["box_list"]
    is_feature_list = tracklet["is_feature"]
    feature_list = tracklet["feature_list"]
    start_frame_id = tracklet["start_frame_id"]
    end_frame_id = tracklet["end_frame_id"]
    start_region_id = tracklet["start_region_id"]
    end_region_id = tracklet["end_region_id"]
    go_through_region = tracklet["go_through_region"]
    if end_frame_id - start_frame_id != len(is_box_list) - 1:
        print("error box number track id:",tracklet["track_id"])
        print("box size {},frame size {}".format(len(is_box_list),(end_frame_id - start_frame_id)))
        raise
    boxes_num = len([i for i in is_box_list if i == 1])
    feature_num = len([i for i in is_feature_list if i == 1])
    if feature_num > boxes_num:
        print("feature box num is error track_id:",tracklet["track_id"])
        raise
    if boxes_num != len(box_list):
        print("boxes_num num is error track_id",tracklet["track_id"])
        raise
    if feature_num != len(feature_list):
        print("feature_num  is error track_id",tracklet["track_id"])
        raise
    if start_region_id != go_through_region[0]:
        print("go_through_region start_region_id  is error track_id",tracklet["track_id"])
        raise
    # if end_region_id != go_through_region[-1]:
    #     print("go_through_region end_region_id  is error track_id",tracklet["track_id"])
    #     raise

def generate_new_tracklet(tracklet_a,tracklet_b,filter_region,cam_name):
    
    tracklet_new = {"track_id":0,"is_box":[],"box_list":[],"is_feature":[],"feature_list":[],
    "start_frame_id":0,"end_frame_id":0,"start_region_id":0,"end_region_id":0,"go_through_region":[]}
    tracklet_new["track_id"] = tracklet_a["track_id"]
    tracklet_new["start_frame_id"] = tracklet_a["start_frame_id"]
    tracklet_new["end_frame_id"] = tracklet_b["end_frame_id"]
    tracklet_new["start_region_id"] = tracklet_a["start_region_id"]
    tracklet_new["end_region_id"] = tracklet_b["end_region_id"]
    tracklet_new["go_through_region"] = tracklet_a["go_through_region"]
    

    is_box_list_b = tracklet_b["is_box"]
    box_list_b = tracklet_b["box_list"]
    is_feature_list_b = tracklet_b["is_feature"]
    feature_list_b = tracklet_b["feature_list"]
    start_frame_id_b = tracklet_b["start_frame_id"]
    
    go_through_region_b = tracklet_b["go_through_region"]
    for region_id in go_through_region_b:
        if region_id not in tracklet_new["go_through_region"]:
            tracklet_new["go_through_region"].append(region_id)

    is_box_list_a = tracklet_a["is_box"]
    box_list_a = tracklet_a["box_list"]
    is_feature_list_a = tracklet_a["is_feature"]
    feature_list_a = tracklet_a["feature_list"]
    start_frame_id_a = tracklet_a["start_frame_id"]
    end_frame_id_a = tracklet_a["end_frame_id"]
    

    have_box_idx_a = 0
    have_feature_idx_a = 0
    have_box_idx_b = 0
    have_feature_idx_b = 0

    offset = tracklet_new["end_frame_id"] - tracklet_new["start_frame_id"]
    print("offset:",offset)
    for idx in range(offset+1):
        frame_id = start_frame_id_a + idx
        #print("frame_id:{},idx:{}".format(frame_id,idx))
        if frame_id >= start_frame_id_b:
            #print("append b")
            idx_ = frame_id - start_frame_id_b
            is_box = is_box_list_b[idx_]
            if is_box == 1:
                box = box_list_b[have_box_idx_b]
                ret = is_in_crowd_region(box,filter_region,cam_name)
                #print("bbret:{},area :{}".format(ret,(box[2]*box[3])))
                if box[2]*box[3] > 500 and ret == False: 
                    tracklet_new["is_box"].append(is_box)
                    tracklet_new["box_list"].append(box_list_b[have_box_idx_b])
                    have_box_idx_b = have_box_idx_b + 1
                    if is_feature_list_b[idx_] == 1:
                        if is_in_crowd_region42(box,filter_region,cam_name)and cam_name =="c042":
                            tracklet_new["is_feature"].append(0)
                            have_feature_idx_b = have_feature_idx_b + 1
                        else:
                            tracklet_new["is_feature"].append(is_feature_list_b[idx_])
                            tracklet_new["feature_list"].append(feature_list_b[have_feature_idx_b])
                            have_feature_idx_b = have_feature_idx_b + 1

                    else:
                        tracklet_new["is_feature"].append(is_feature_list_b[idx_])
                else:
                    have_box_idx_b = have_box_idx_b + 1
                    tracklet_new["is_box"].append(0)
                    tracklet_new["is_feature"].append(0)
            else:
                tracklet_new["is_box"].append(is_box)
                if is_feature_list_b[idx_] == 1:
                    print("*********error")
                    raise
                    tracklet_new["is_feature"].append(is_feature_list_b[idx_])
                    tracklet_new["feature_list"].append(feature_list_b[have_feature_idx_b])
                    have_feature_idx_b = have_feature_idx_b + 1
                else:
                    tracklet_new["is_feature"].append(is_feature_list_b[idx_])
            continue
        if  frame_id > end_frame_id_a and frame_id < start_frame_id_b:
            tracklet_new["is_box"].append(0)
            tracklet_new["is_feature"].append(0)
        if frame_id <= end_frame_id_a:
            #print("append a")
            p = [i for i in is_feature_list_a if i == 1]
            is_box = is_box_list_a[idx]
            if is_box == 1:
                box = box_list_a[have_box_idx_a]
                ret = is_in_crowd_region(box,filter_region,cam_name)
                #print("aaret:{},area :{}".format(ret,(box[2]*box[3])))
                if box[2]*box[3] > 500 and ret == False: 
                    tracklet_new["box_list"].append(box_list_a[have_box_idx_a])
                    tracklet_new["is_box"].append(is_box)
                    have_box_idx_a = have_box_idx_a + 1
                    if is_feature_list_a[idx] == 1:
                        aa = is_in_crowd_region42(box,filter_region,cam_name)
                        #print("have_feature_idx_a:",aa)
                        if aa and cam_name =="c042":
                            tracklet_new["is_feature"].append(0)
                            have_feature_idx_a = have_feature_idx_a + 1
                        else:
                            tracklet_new["is_feature"].append(is_feature_list_a[idx])
                            tracklet_new["feature_list"].append(feature_list_a[have_feature_idx_a])
                            have_feature_idx_a = have_feature_idx_a + 1

                    else:
                        tracklet_new["is_feature"].append(is_feature_list_a[idx])
                else:
                    have_box_idx_a = have_box_idx_a + 1
                    tracklet_new["is_box"].append(0)
                    tracklet_new["is_feature"].append(0)
            else:
                tracklet_new["is_box"].append(is_box)
                if is_feature_list_a[idx] == 1:
                    print("*********error")
                    raise
                    tracklet_new["is_feature"].append(is_feature_list_a[idx])
                    tracklet_new["feature_list"].append(feature_list_a[have_feature_idx_a])
                    have_feature_idx_a = have_feature_idx_a + 1
                else:
                    tracklet_new["is_feature"].append(is_feature_list_a[idx])
    check_tracklet(tracklet_new)
    return tracklet_new
    
         
def function(date):
    return date['start_frame_id']
 
def merger_tracklet(track_id_list,tracklets,filter_region,cam_name):
    #tracklets = copy.deepcopy(tracklets_)
    candidate = []
    for track_id in track_id_list:
        for tracklet in tracklets:
            if track_id == tracklet["track_id"]:
                candidate.append(tracklet)
    candidate.sort(key=function)
    if len(candidate) < 2:
        print("error candidate")
        raise
    tracklet_new = {}
    
    for indx in range(len(candidate)):
        current_tracklet = candidate[indx]
        if indx == 0:
            tracklet_new = candidate[indx]
            continue
        else:
            tracklet_new = generate_new_tracklet(tracklet_new,current_tracklet,filter_region,cam_name)
    return tracklet_new


def joint_tracklets(tracklets,track_id_list,cam_name,base_path,filter_region):#tracklet_0,cam_name):
    
    cost_matrix = get_cost_matrix(tracklets)
    #print("cost_matrix:",cost_matrix)
    score = 0.0
    # if cam_name == "c042":
    #     score = 0.7
    # else:
    #     score = 0.7

    cluster_labels = AgglomerativeClustering(n_clusters=None, distance_threshold=0.7, affinity='precomputed',
                                linkage='complete').fit_predict(cost_matrix)
    print("cluster_labels:",type(cluster_labels))
    print("cluster_labels:",cluster_labels)
    labels = get_match(cluster_labels)
    print("labels:",labels)
    for tracket_conn in labels:
        connection_tracklet = []
        for idx in tracket_conn:
            connection_tracklet.append(tracklets[idx])
        print("******")
        for track_i in tracket_conn:
            print("track_id:",track_id_list[track_i])
        keep_track_id,remove_track_id = check_tracklets(connection_tracklet)
        print("keep_tracklet_id:",keep_track_id)
        print("remove_track_id:",remove_track_id)

        if len(keep_track_id) >= 2:
            res = merger_tracklet(keep_track_id,tracklets,filter_region,cam_name)
            new_tracklet_path = os.path.join(base_path,cam_name+"/connect/")
            new_tracklet_file = os.path.join(new_tracklet_path,str(res["track_id"]) + ".json")
            jsObj = json.dumps(res)    
            if not os.path.exists(new_tracklet_path):
                os.makedirs(new_tracklet_path)
            with open(new_tracklet_file, "w") as f:  
                f.write(jsObj)  
                f.close() 
            move_joint_tracklet(keep_track_id,cam_name,base_path,"/remove/",res["track_id"])
        if  len(keep_track_id) == 1:
            pass
        if len(remove_track_id) >= 2 and len(keep_track_id) >= 1:
            while True:
                conn_tracklets = []
                for track_id in remove_track_id:
                    for tracklet in tracklets:
                        if track_id == tracklet["track_id"]:
                            conn_tracklets.append(tracklet)
                keep_track_id,remove_track_id = check_tracklets(conn_tracklets)
                if (len(keep_track_id) >=2):
                    res = merger_tracklet(keep_track_id,tracklets,filter_region,cam_name)
                    new_tracklet_path = os.path.join(base_path,cam_name+"/connect/")
                    new_tracklet_file = os.path.join(new_tracklet_path,str(res["track_id"]) + ".json")
                    jsObj = json.dumps(res)    
                    if not os.path.exists(new_tracklet_path):
                        os.makedirs(new_tracklet_path)
                    with open(new_tracklet_file, "w") as f:  
                        f.write(jsObj)  
                        f.close()
                    move_joint_tracklet(keep_track_id,cam_name,base_path,"/remove/",res["track_id"])
                print("keep_tracklet_id:",keep_track_id)
                print("remove_track_id:",remove_track_id)
                if(len(remove_track_id) < 2) or (len(keep_track_id) < 2):
                    break
                
def move_joint_tracklet(remove_tracklets,cam_name,base_path,type_remove,merger_track_id):
    print("move_joint_tracklet cam {} move {} type {}".format(cam_name,len(remove_tracklets),type_remove))
    print("move list:",remove_tracklets)
    srcpath = os.path.join(base_path,cam_name+"/result/")
    despath = os.path.join(base_path,cam_name+type_remove)
    for track_id in remove_tracklets:
        srcfile = os.path.join(srcpath,str(track_id) + ".json")
        desfile = os.path.join(despath,str(merger_track_id) +"-"+str(track_id)+ ".json")
        mymovefile(srcfile,desfile)

def cam_joint_tracklets(all_tracklet,new_all_mask,reconn_regions,base_path):
    filter_region = load_reconn_filter_region("../re_region")
    for cam_name in all_tracklet:
        if cam_name == "c041":
            middle_regions = [10]
            start_region_ids = [1,3,5,7]
            end_region_ids = [2,4,6,8]
            tracklets = []
            for tracklet in all_tracklet[cam_name]:
                feature_array = np.array(tracklet["feature_list"])
                avg_feature = np.mean(feature_array,axis=0)
                tracklet['avg_feature'] = avg_feature.tolist()
                start_region_id = tracklet["start_region_id"]
                end_region_id = tracklet["end_region_id"]
                start_frame_id = tracklet["start_frame_id"]
                end_frame_id = tracklet["end_frame_id"]
                if end_region_id in start_region_ids:   #在开始区域结束
                    if end_frame_id != 2000:
                        if cal_boxes_avg_area(tracklet["box_list"]) > 700:
                            tracklets.append(tracklet)
                            continue
                if end_region_id in middle_regions:   #在中间区域结束
                    if end_frame_id != 2000:
                        if cal_boxes_avg_area(tracklet["box_list"]) > 700:
                            tracklets.append(tracklet)
                            continue
                if(end_region_id in end_region_ids) and (start_region_id not in start_region_ids):#不在开始区域创建
                    if start_frame_id != 0 and end_frame_id != 2000:
                        if cal_boxes_avg_area(tracklet["box_list"]) > 700:
                            tracklets.append(tracklet)
                            continue
                if start_frame_id != 0:         #第一个框在中间区域
                    box = tracklet["box_list"][0]
                    if is_interaction_middle_region(box,cam_name,new_all_mask):
                        if cal_boxes_avg_area(tracklet["box_list"]) > 700:
                            tracklets.append(tracklet)
                            continue
                if is_in_reconnect_region(tracklet["box_list"][0],reconn_regions,cam_name):  #在自定义拼接区域
                    if cal_boxes_avg_area(tracklet["box_list"]) > 800:
                        if end_region_id == 6 or end_region_id == 4:
                            continue
                        tracklets.append(tracklet)
                        continue
            track_id_list = []
            for tracklet in tracklets:
                track_id_list.append(tracklet["track_id"])
            print("exception tracklet len:",len(track_id_list))
            print("exception tracklet:",track_id_list)
            joint_tracklets(tracklets,track_id_list,cam_name,base_path,filter_region)
            copy_joint_tracklet(cam_name,base_path,"/connect/","/result/")

        if cam_name == "c042":
            middle_regions = [10]
            start_region_ids = [1,3,5,7]
            end_region_ids = [2,4,6,8]
            tracklets = []
            for tracklet in all_tracklet[cam_name]:
                feature_array = np.array(tracklet["feature_list"])
                avg_feature = np.mean(feature_array,axis=0)
                tracklet['avg_feature'] = avg_feature.tolist()
                start_region_id = tracklet["start_region_id"]
                end_region_id = tracklet["end_region_id"]
                start_frame_id = tracklet["start_frame_id"]
                end_frame_id = tracklet["end_frame_id"]
                # if end_region_id in start_region_ids:   #在开始区域结束
                #     if end_frame_id != 2000:
                #         if cal_boxes_avg_area(tracklet["box_list"]) > 700:
                #             tracklets.append(tracklet)
                #             continue
                if end_region_id in middle_regions:   #在中间区域结束
                    if end_frame_id != 2000:
                        if cal_boxes_avg_area(tracklet["box_list"]) > 700:
                            tracklets.append(tracklet)
                            continue
                if(end_region_id in end_region_ids) and (start_region_id not in start_region_ids):#不在开始区域创建
                    if start_frame_id != 0 and end_frame_id != 2000:
                        if cal_boxes_avg_area(tracklet["box_list"]) > 700:
                            tracklets.append(tracklet)
                            continue
                if is_in_reconnect_region(tracklet["box_list"][0],reconn_regions,cam_name):  #在自定义拼接区域
                    if cal_boxes_avg_area(tracklet["box_list"]) > 800:
                        tracklets.append(tracklet)
                        continue
            track_id_list = []
            for tracklet in tracklets:
                track_id_list.append(tracklet["track_id"])
            print("exception tracklet len:",len(track_id_list))
            print("exception tracklet:",track_id_list)
            joint_tracklets(tracklets,track_id_list,cam_name,base_path,filter_region)
            copy_joint_tracklet(cam_name,base_path,"/connect/","/result/")
        if cam_name == "c043":
            middle_regions = [10]
            start_region_ids = [1,3,5,7]
            end_region_ids = [2,4,6,8]
            tracklets = []
            for tracklet in all_tracklet[cam_name]:
                feature_array = np.array(tracklet["feature_list"])
                avg_feature = np.mean(feature_array,axis=0)
                tracklet['avg_feature'] = avg_feature.tolist()
                start_region_id = tracklet["start_region_id"]
                end_region_id = tracklet["end_region_id"]
                start_frame_id = tracklet["start_frame_id"]
                end_frame_id = tracklet["end_frame_id"]
                if end_region_id in start_region_ids:   #在开始区域结束
                    if end_frame_id != 2000:
                        if cal_boxes_avg_area(tracklet["box_list"]) > 700:
                            tracklets.append(tracklet)
                            continue
                if end_region_id in middle_regions:   #在中间区域结束
                    if end_frame_id != 2000:
                        if cal_boxes_avg_area(tracklet["box_list"]) > 700:
                            tracklets.append(tracklet)
                            continue
                if(end_region_id in end_region_ids) and (start_region_id not in start_region_ids):#不在开始区域创建
                    if start_frame_id != 0 and end_frame_id != 2000:
                        if cal_boxes_avg_area(tracklet["box_list"]) > 700:
                            tracklets.append(tracklet)
                            continue
                if start_frame_id != 0:         #第一个框在中间区域
                    box = tracklet["box_list"][0]
                    if is_interaction_middle_region(box,cam_name,new_all_mask):
                        if cal_boxes_avg_area(tracklet["box_list"]) > 700:
                            tracklets.append(tracklet)
                            continue
                if is_in_reconnect_region(tracklet["box_list"][0],reconn_regions,cam_name):  #在自定义拼接区域
                    if cal_boxes_avg_area(tracklet["box_list"]) > 800:
                        if end_region_id == 6 or end_region_id == 4:
                            continue
                        tracklets.append(tracklet)
                        continue
            track_id_list = []
            for tracklet in tracklets:
                track_id_list.append(tracklet["track_id"])
            print("exception tracklet len:",len(track_id_list))
            print("exception tracklet:",track_id_list)
            joint_tracklets(tracklets,track_id_list,cam_name,base_path,filter_region)
            copy_joint_tracklet(cam_name,base_path,"/connect/","/result/")
        if cam_name == "c044":
            middle_regions = [10]
            start_region_ids = [1,3,5,7]
            end_region_ids = [2,4,6,8]
            tracklets = []
            for tracklet in all_tracklet[cam_name]:
                feature_array = np.array(tracklet["feature_list"])
                avg_feature = np.mean(feature_array,axis=0)
                tracklet['avg_feature'] = avg_feature.tolist()
                start_region_id = tracklet["start_region_id"]
                end_region_id = tracklet["end_region_id"]
                start_frame_id = tracklet["start_frame_id"]
                end_frame_id = tracklet["end_frame_id"]
                if end_region_id in start_region_ids:   #在开始区域结束
                    if end_frame_id != 2000:
                        if cal_boxes_avg_area(tracklet["box_list"]) > 700:
                            tracklets.append(tracklet)
                            continue
                if end_region_id in middle_regions:   #在中间区域结束
                    if end_frame_id != 2000:
                        if cal_boxes_avg_area(tracklet["box_list"]) > 700:
                            tracklets.append(tracklet)
                            continue
                if(end_region_id in end_region_ids) and (start_region_id not in start_region_ids):#不在开始区域创建
                    if start_frame_id != 0 and end_frame_id != 2000:
                        if cal_boxes_avg_area(tracklet["box_list"]) > 700:
                            tracklets.append(tracklet)
                            continue
                if start_frame_id != 0:         #第一个框在中间区域
                    box = tracklet["box_list"][0]
                    if is_interaction_middle_region(box,cam_name,new_all_mask):
                        if cal_boxes_avg_area(tracklet["box_list"]) > 700:
                            tracklets.append(tracklet)
                            continue
                # if is_in_reconnect_region(tracklet["box_list"][0],reconn_regions,cam_name):  #在自定义拼接区域
                #     if cal_boxes_avg_area(tracklet["box_list"]) > 800:
                #         if end_region_id == 6 or end_region_id == 4:
                #             continue
                #         tracklets.append(tracklet)
                #         continue
            track_id_list = []
            for tracklet in tracklets:
                track_id_list.append(tracklet["track_id"])
            print("exception tracklet len:",len(track_id_list))
            print("exception tracklet:",track_id_list)
            joint_tracklets(tracklets,track_id_list,cam_name,base_path,filter_region)
            copy_joint_tracklet(cam_name,base_path,"/connect/","/result/")
        if cam_name == "c045":
            middle_regions = [10]
            start_region_ids = [1,3,5,7]
            end_region_ids = [2,4,6,8]
            tracklets = []
            for tracklet in all_tracklet[cam_name]:
                feature_array = np.array(tracklet["feature_list"])
                avg_feature = np.mean(feature_array,axis=0)
                tracklet['avg_feature'] = avg_feature.tolist()
                start_region_id = tracklet["start_region_id"]
                end_region_id = tracklet["end_region_id"]
                start_frame_id = tracklet["start_frame_id"]
                end_frame_id = tracklet["end_frame_id"]
                if end_region_id in start_region_ids:   #在开始区域结束
                    if end_frame_id != 2000:
                        if cal_boxes_avg_area(tracklet["box_list"]) > 700:
                            tracklets.append(tracklet)
                            continue
                if end_region_id in middle_regions:   #在中间区域结束
                    if end_frame_id != 2000:
                        if cal_boxes_avg_area(tracklet["box_list"]) > 700:
                            tracklets.append(tracklet)
                            continue
                if(end_region_id in end_region_ids) and (start_region_id not in start_region_ids):#不在开始区域创建
                    if start_frame_id != 0 and end_frame_id != 2000:
                        if cal_boxes_avg_area(tracklet["box_list"]) > 700:
                            tracklets.append(tracklet)
                            continue
                if start_frame_id != 0:         #第一个框在中间区域
                    box = tracklet["box_list"][0]
                    if is_interaction_middle_region(box,cam_name,new_all_mask):
                        if cal_boxes_avg_area(tracklet["box_list"]) > 700:
                            tracklets.append(tracklet)
                            continue
                # if is_in_reconnect_region(tracklet["box_list"][0],reconn_regions,cam_name):  #在自定义拼接区域
                #     if cal_boxes_avg_area(tracklet["box_list"]) > 800:
                #         if end_region_id == 6 or end_region_id == 4:
                #             continue
                #         tracklets.append(tracklet)
                #         continue
            track_id_list = []
            for tracklet in tracklets:
                track_id_list.append(tracklet["track_id"])
            print("exception tracklet len:",len(track_id_list))
            print("exception tracklet:",track_id_list)
            joint_tracklets(tracklets,track_id_list,cam_name,base_path,filter_region)
            copy_joint_tracklet(cam_name,base_path,"/connect/","/result/")
        if cam_name == "c046":
            middle_regions = [10]
            start_region_ids = [1,3,5,7]
            end_region_ids = [2,4,6,8]
            tracklets = []
            for tracklet in all_tracklet[cam_name]:
                feature_array = np.array(tracklet["feature_list"])
                avg_feature = np.mean(feature_array,axis=0)
                tracklet['avg_feature'] = avg_feature.tolist()
                start_region_id = tracklet["start_region_id"]
                end_region_id = tracklet["end_region_id"]
                start_frame_id = tracklet["start_frame_id"]
                end_frame_id = tracklet["end_frame_id"]
                if end_region_id in start_region_ids:   #在开始区域结束
                    if end_frame_id != 2000:
                        if cal_boxes_avg_area(tracklet["box_list"]) > 700:
                            tracklets.append(tracklet)
                            continue
                if end_region_id in middle_regions:   #在中间区域结束
                    if end_frame_id != 2000:
                        if cal_boxes_avg_area(tracklet["box_list"]) > 700:
                            tracklets.append(tracklet)
                            continue
                if(end_region_id in end_region_ids) and (start_region_id not in start_region_ids):#不在开始区域创建
                    if start_frame_id != 0 and end_frame_id != 2000:
                        if cal_boxes_avg_area(tracklet["box_list"]) > 700:
                            tracklets.append(tracklet)
                            continue
                if start_frame_id != 0:         #第一个框在中间区域
                    box = tracklet["box_list"][0]
                    if is_interaction_middle_region(box,cam_name,new_all_mask):
                        if cal_boxes_avg_area(tracklet["box_list"]) > 700:
                            tracklets.append(tracklet)
                            continue
                # if is_in_reconnect_region(tracklet["box_list"][0],reconn_regions,cam_name):  #在自定义拼接区域
                #     if cal_boxes_avg_area(tracklet["box_list"]) > 800:
                #         if end_region_id == 6 or end_region_id == 4:
                #             continue
                #         tracklets.append(tracklet)
                #         continue
            track_id_list = []
            for tracklet in tracklets:
                track_id_list.append(tracklet["track_id"])
            print("exception tracklet len:",len(track_id_list))
            print("exception tracklet:",track_id_list)
            joint_tracklets(tracklets,track_id_list,cam_name,base_path,filter_region)
            copy_joint_tracklet(cam_name,base_path,"/connect/","/result/")