import shutil
import os
import cv2
import numpy as np
import json
import math

def set_dir(filepath):
    if not os.path.exists(filepath):
        os.mkdir(filepath)
    else:
        shutil.rmtree(filepath)
        os.mkdir(filepath)

def load_tracklet(track_path):
    cam_tracklet = []
    tracklet_files = os.listdir(track_path)
    for tracklet in tracklet_files:
        file_name = os.path.join(track_path,tracklet)
        if "gitkeep" in file_name:
            os.remove(file_name)
            continue
        with open(file_name,'r',encoding='utf-8') as f_obj:
            jsObj = json.load(f_obj)
            if type(jsObj) == dict:
                if "track_id" in jsObj:
                    cam_tracklet.append(jsObj)
                else:
                    print("key track id is not in dict",file_name)
                    raise
            else:
                print("jsObj is not dict ",file_name)
    return cam_tracklet

def load_mask(base_path):
    new_all_mask = {"c041":{},"c042":{},"c043":{},"c044":{},"c045":{},"c046":{}}
    cam_name_list = ["c041","c042","c043","c044","c045","c046"]
    for cam_name in cam_name_list:
        mask_path = os.path.join(base_path,cam_name)
        mask_files = os.listdir(mask_path)
        mask_dict = {}
        for file in mask_files:
            mask_path_file = os.path.join(mask_path,file)
            file_name = file.split(".")[0]
            mask_dict[file_name] = cv2.imread(mask_path_file,cv2.IMREAD_GRAYSCALE)
        new_all_mask[cam_name] = mask_dict
    return new_all_mask

def load_reconn_region(json_path):
    cam_name_list = ["c041","c042","c043","c044","c045","c046"]
    cam_frame_size = {"c041":(1280,960,3),"c042":(1280,960,3),"c043":(1280,960,3),
    "c044":(1280,960,3),"c045":(1280,720,3),"c046":(1280,720,3)}
    reconn_regions = {} 
    for cam_name in cam_name_list:
        json_file = os.path.join(json_path,cam_name+".json")
        new_dict = {}
        with open(json_file, 'r') as f:
            new_dict = json.load(f)
        shapes = new_dict["shapes"]
        image_size = cam_frame_size[cam_name]
        for shape in shapes:
            if shape["label"] == "10":
                region_points = shape["points"]
                new_points = []
                points_ = []
                for region_point in region_points:
                    region_point_i = [int(region_point[0]),int(region_point[1])]
                    new_points.append(region_point_i)
                img_zero = np.zeros((image_size[1],image_size[0]), dtype=np.uint8)
                points_.append(new_points)
                cv2.fillPoly(img_zero,np.array([new_points]),(255))
                reconn_regions[cam_name] = img_zero
    return reconn_regions

def load_reconn_filter_region(json_path):
    cam_name_list = ["c041","c042","c043","c044","c045","c046"]
    cam_frame_size = {"c041":(1280,960,3),"c042":(1280,960,3),"c043":(1280,960,3),
    "c044":(1280,960,3),"c045":(1280,720,3),"c046":(1280,720,3)}
    reconn_regions = {} 
    for cam_name in cam_name_list:
        json_file = os.path.join(json_path,cam_name+".json")
        new_dict = {}
        with open(json_file, 'r') as f:
            new_dict = json.load(f)
        shapes = new_dict["shapes"]
        image_size = cam_frame_size[cam_name]
        for shape in shapes:
            if shape["label"] == "1":
                region_points = shape["points"]
                new_points = []
                points_ = []
                for region_point in region_points:
                    region_point_i = [int(region_point[0]),int(region_point[1])]
                    new_points.append(region_point_i)
                img_zero = np.zeros((image_size[1],image_size[0]), dtype=np.uint8)
                points_.append(new_points)
                cv2.fillPoly(img_zero,np.array([new_points]),(255))
                reconn_regions[cam_name] = img_zero
                # cv2.imshow(cam_name,img_zero)
                # cv2.waitKey(10000)
    return reconn_regions



def is_interaction_middle_region(box,cam_name,new_all_mask):
    cam_frame_size = {"c041":(1280,960,3),"c042":(1280,960,3),"c043":(1280,960,3),
    "c044":(1280,960,3),"c045":(1280,720,3),"c046":(1280,720,3)}
    middle_mask = new_all_mask[cam_name]["10"]
    image_size = cam_frame_size[cam_name]
    img_zero = np.zeros((image_size[1],image_size[0]), dtype=np.uint8)
    cv2.rectangle(img_zero,(box[0],box[1]),(box[0]+box[2],box[1]+box[3]),(255),-1)
    A_and_B = img_zero & middle_mask
    index = np.where(A_and_B > 0)
    area = index[0].shape[0]
    if(area > 4):
        return True
    else:
        return False

def cal_boxes_avg_area(boxes):
    total_eare = 0.0
    for box in boxes:
        area = box[2]*box[3]
        total_eare = total_eare + area
    avg_area = total_eare/len(boxes)
    return avg_area

def is_in_reconnect_region(box,reconn_regions,cam_name):
    cam_frame_size = {"c041":(1280,960,3),"c042":(1280,960,3),"c043":(1280,960,3),
    "c044":(1280,960,3),"c045":(1280,720,3),"c046":(1280,720,3)}
    image_size = cam_frame_size[cam_name]
    img_zero = np.zeros((image_size[1],image_size[0]), dtype=np.uint8)
    cv2.rectangle(img_zero,(box[0],box[1]),(box[0]+box[2],box[1]+box[3]),(255),-1)
    A_and_B = img_zero & reconn_regions[cam_name]
    index = np.where(A_and_B > 0)
    area = index[0].shape[0]
    if(area > 10):
        return True
    else:
        return False

def is_in_crowd_region(box,filter_regions,cam_name):
    if cam_name != "c041":
        return False
    if cam_name not in filter_regions:
        return False
    cam_frame_size = {"c041":(1280,960,3),"c042":(1280,960,3),"c043":(1280,960,3),
    "c044":(1280,960,3),"c045":(1280,720,3),"c046":(1280,720,3)}
    image_size = cam_frame_size[cam_name]
    img_zero = np.zeros((image_size[1],image_size[0]), dtype=np.uint8)
    cv2.rectangle(img_zero,(box[0],box[1]),(box[0]+box[2],box[1]+box[3]),(255),-1)
    A_and_B = img_zero & filter_regions[cam_name]
    index = np.where(A_and_B > 0)
    area = index[0].shape[0]
    if(area > 10):
        return True
    else:
        return False

def is_in_crowd_region42(box,filter_regions,cam_name):
    return False
    # if cam_name != "c041" or cam_name != "c042":
    #     print("cam_name != ",cam_name)
    #     return False
    if cam_name not in filter_regions:
        print("cam_name not in filter_regions")
        return False
    cam_frame_size = {"c041":(1280,960,3),"c042":(1280,960,3),"c043":(1280,960,3),
    "c044":(1280,960,3),"c045":(1280,720,3),"c046":(1280,720,3)}
    image_size = cam_frame_size[cam_name]
    img_zero = np.zeros((image_size[1],image_size[0]), dtype=np.uint8)
    cv2.rectangle(img_zero,(box[0],box[1]),(box[0]+box[2],box[1]+box[3]),(255),-1)
    # cv2.imshow("123",filter_regions[cam_name])
    # cv2.waitKey(100000)
    A_and_B = img_zero & filter_regions[cam_name]
    index = np.where(A_and_B > 0)
    area = index[0].shape[0]
    if(area > 10):
        return True
    else:
        return False


def IoU(box1, box2):
    """
    :param box1: list in format [xmin1, ymin1, xmax1, ymax1]
    :param box2:  list in format [xmin2, ymin2, xamx2, ymax2]
    :return:    returns IoU ratio (intersection over union) of two boxes
    """
    xmin1, ymin1, xmax1, ymax1 = box1
    xmin2, ymin2, xmax2, ymax2 = box2
    x_overlap = max(0, min(xmax1, xmax2) - max(xmin1, xmin2))
    y_overlap = max(0, min(ymax1, ymax2) - max(ymin1, ymin2))
    intersection = x_overlap * y_overlap
    union = (xmax1 - xmin1) * (ymax1 - ymin1) + (xmax2 - xmin2) * (ymax2 - ymin2) - intersection
    return float(intersection) / union

def cal_min_iou(boxes):
    new_boxes = []
    for box in boxes:
        new_box = [box[0],box[1],box[0] +box[2] ,box[1]+box[3]]
        new_boxes.append(new_box)
    min_iou_score = 1.0
    box_length = len(boxes)
    for i,box1 in enumerate(new_boxes):
        if i == box_length - 1:
            return min_iou_score
        else:
            for j,box2 in enumerate(new_boxes):
                if i < j:
                    current_iou = IoU(box1,box2)
                    if current_iou == 0.0:
                        return 0.0
                    if current_iou < min_iou_score:
                        min_iou_score = current_iou
    return min_iou_score

def distance_boxes(box1,box2):
    center_x_0 = box1[0] + box1[2]/2
    center_y_0 = box1[1] + box1[3]/2
    center_x_1 = box2[0] + box2[2]/2
    center_y_1 = box2[1] + box2[3]/2

    distance_b  = math.sqrt(math.pow((center_x_1-center_x_0),2)+math.pow((center_y_1-center_y_0),2))
    return distance_b

def distance_bet_two(boxes_list):
    if(len(boxes_list) < 2):
        print("boxes_list is less than 2")
        raise
    max_dis = 0
    for i,box in enumerate(boxes_list):
        j = i + 1
        if j == len(boxes_list) -1:
            dis = distance_boxes(boxes_list[i],boxes_list[j])
            if dis > max_dis:
                max_dis = dis
            return max_dis
        else:
            dis = distance_boxes(boxes_list[i],boxes_list[j])
            if dis > max_dis:
                max_dis = dis
    print("error")
    raise

def is_interaction_out(boxes,cam_name,new_all_mask):
    cam_frame_size = {"c041":(1280,960,3),"c042":(1280,960,3),"c043":(1280,960,3),
    "c044":(1280,960,3),"c045":(1280,720,3),"c046":(1280,720,3)}
    image_size = cam_frame_size[cam_name]
    out_box_num = 0
    for box in boxes:
        img_zero = np.zeros((image_size[1],image_size[0]), dtype=np.uint8)
        cv2.rectangle(img_zero,(box[0],box[1]),(box[0]+box[2],box[1]+box[3]),(255),-1)
        A_and_B = img_zero & new_all_mask[cam_name]["unmask"]
        index = np.where(A_and_B > 0)
        area = index[0].shape[0]
        if(area > 10):
            out_box_num = out_box_num+ 1
    rotia = out_box_num*1.0/len(boxes)*1.0
    if(rotia > 0.8):
        return True
    else:
        return False

def mymovefile(srcfile,dstfile):
    if not os.path.isfile(srcfile):
        print ("%s not exist!"%(srcfile))
    else:
        fpath,fname=os.path.split(dstfile)    #分离文件名和路径
        if not os.path.exists(fpath):
            os.makedirs(fpath)                #创建路径
        shutil.move(srcfile,dstfile)          #移动文件
        #print ("move %s -> %s"%( srcfile,dstfile))

def move_tracklet(remove_tracklets,cam_name,base_path,type_remove):
    print("cam {} move {} type {}".format(cam_name,len(remove_tracklets),type_remove))
    print("move list:",remove_tracklets)
    srcpath = os.path.join(base_path,cam_name+"/result/")
    despath = os.path.join(base_path,cam_name+type_remove)
    for key in remove_tracklets:
        for track_id in remove_tracklets[key]:
            srcfile = os.path.join(srcpath,str(track_id) + ".json")
            desfile = os.path.join(despath,str(track_id) + ".json")
            mymovefile(srcfile,desfile)

def mycopyfile(srcfile,dstfile):
    if not os.path.isfile(srcfile):
        print ("%s not exist!"%(srcfile))
    else:
        fpath,fname=os.path.split(dstfile)    
        if not os.path.exists(fpath):
            os.makedirs(fpath)                
        shutil.copyfile(srcfile,dstfile)      
        #print ("copy %s -> %s"%( srcfile,dstfile))

def copy_joint_tracklet(cam_name,base_path,src_type,des_type):
    
    srcpath = os.path.join(base_path,cam_name+src_type)
    despath = os.path.join(base_path,cam_name+des_type)
    if not os.path.exists(despath):
        os.makedirs(despath)
    files = os.listdir(srcpath)
    for tracklet_file in files:
        src_file = os.path.join(srcpath,tracklet_file)
        des_file = os.path.join(despath,tracklet_file)
        if os.path.exists(des_file):
            print("{} is exist".format(des_file))
            raise
        mycopyfile(src_file,des_file)

def move_keep_tracklet(keep_tracklet_list,cam_name,base_path,src_type,des_type):
    srcpath = os.path.join(base_path,cam_name+src_type)
    despath = os.path.join(base_path,cam_name+des_type)
    for track_id in keep_tracklet_list:
        src_file = os.path.join(srcpath,str(track_id)+".json")
        des_file = os.path.join(despath,str(track_id)+".json")
        if os.path.exists(des_file):
            print("{} is exist".format(des_file))
            raise
        mymovefile(src_file,des_file)

def move_time_tracklet(keep_tracklet_list,cam_name,base_path,src_type,des_type):
    srcpath = os.path.join(base_path,cam_name+src_type)
    despath = os.path.join(base_path,cam_name+des_type)
    for track_id in keep_tracklet_list:
        src_file = os.path.join(srcpath,str(track_id)+".json")
        des_file = os.path.join(despath,str(track_id)+".json")
        if os.path.exists(des_file):
            print("{} is exist".format(des_file))
            raise
        mymovefile(src_file,des_file)


