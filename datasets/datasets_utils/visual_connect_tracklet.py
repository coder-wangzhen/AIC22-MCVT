import cv2
import json
import os


def load_tracklet(track_path):
    cam_tracklet = []
    tracklet_files = os.listdir(track_path)
    for tracklet in tracklet_files:
        file_name = os.path.join(track_path,tracklet)
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
def function(date):
    return date['start_frame_id']

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
    if start_region_id not in [1,3,5,7,10]:
        print("start_region_id  is error track_id",tracklet["track_id"])
        raise
    # if end_region_id not in [2,4,6,8]:
    #     print("end_region_id  is error track_id",tracklet["track_id"])
    #     raise
    if start_region_id != go_through_region[0]:
        print("go_through_region start_region_id  is error track_id",tracklet["track_id"])
        raise
    if end_region_id != go_through_region[-1]:
        print("go_through_region end_region_id  is error track_id",tracklet["track_id"])
        raise
def get_boxes(frame_id,tracklets_frame_box):
    track_id_boxes = {}
    for track_id in tracklets_frame_box:
        frame_box = tracklets_frame_box[track_id]
        if frame_id in frame_box.keys():
            track_id_boxes[track_id] = frame_box[frame_id]
    return track_id_boxes
def get_number(data):
    return int(data.split(".")[0])
    
image_base_path = "../../image"
save_image_path = "../../image_save"
base_path = "/mnt/LocalDisk1/Projects/AIC21-MTMC/datasets/algorithm_results/detect_merge"
cam_name_list = ["c041","c042","c043","c044","c045","c046"]
#visial_cam = ["c041","c042","c043","c044","c045","c046"]
visial_cam = ["c042"]
if __name__=='__main__':
    for cam_name in cam_name_list:
        if cam_name not in visial_cam:
            continue
        track_path = os.path.join(base_path,cam_name+"/test/")
        tracklets = load_tracklet(track_path)
        
        if(len(tracklets) == 0):
            print("len(tracklets) == 0")
            raise
        print("tracklets len:",len(tracklets))
        tracklets.sort(key=function)
        tracklets_frame_box = {}
        for tracklet in tracklets:
            print("check track id:",tracklet["track_id"])
            check_tracklet(tracklet)
            start_frame_id = tracklet["start_frame_id"]
            box_list = tracklet["box_list"]
            track_id = tracklet["track_id"]
            have_box_id = 0
            frame_box = {}
            for idx,is_box in enumerate(tracklet["is_box"]):
                current_frame_id = idx + start_frame_id
                if is_box == 1:
                    frame_box[current_frame_id] = box_list[have_box_id]
                    have_box_id = have_box_id + 1
                    
            tracklets_frame_box[track_id] = frame_box
        save_image_file = os.path.join(save_image_path,cam_name+"/connect_image/")
        image_path =  os.path.join(image_base_path,cam_name)
        image_files = os.listdir(image_path)
        image_files.sort(key=get_number)
        for frame_id,file in enumerate(image_files):
            print("before get_boxes")
            track_id_boxes = get_boxes(frame_id,tracklets_frame_box)
            print("after get_boxes")
            if track_id_boxes is None:
                print("current_frame not box frame id:{}".format(frame_id) )
                #continue
            image_path_file = os.path.join(image_path,file)
            print("image_path_file:",image_path_file)
            image = cv2.imread(image_path_file)
            if image is None:
                print("Fail to read image")
                raise
            print("read image")
            for track_id in track_id_boxes:
                box = track_id_boxes[track_id]
                cv2.rectangle(image, (box[0], box[1]), (box[0] + box[2], box[1]+box[3]), (0, 255, 0), 2)
                cv2.putText(image, str(track_id), (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            save_image_file_temp = os.path.join(save_image_file,str(frame_id) + ".jpg")
            print("save_image_file:",save_image_file_temp)
            cv2.imwrite(save_image_file_temp,image) 




    

