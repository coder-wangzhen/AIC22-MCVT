import os
import cv2

def get_result(file_path):
    result = {"41":{},"42":{},"43":{},"44":{},"45":{},"46":{}}
    with open(file_path,'r',encoding='utf-8') as f_obj:
        txt_lines = f_obj.readlines()
        result_frame = {}
        box_list = []
        pre_frame_id = "1"
        pre_cam_id = "41"
        for line_idx,txt_line in enumerate(txt_lines):
            txt_line = txt_line.rstrip("\n")
            if len(txt_line) < 4:
                if line_idx == len(txt_lines) -1:
                    result_frame[pre_frame_id] = box_list
                    result[pre_cam_id] = result_frame
                    result_frame = {}
                    return result
            txt_list = txt_line.split(" ")
            frame_id = txt_list[2]
            cam_id = txt_list[0]
            # print("cam_id:",cam_id)
            # print("frame_id:",frame_id)
            if line_idx == len(txt_lines) -1:
                result_frame[pre_frame_id] = box_list
                result[pre_cam_id] = result_frame
                result_frame = {}
                return result
            if pre_cam_id != cam_id:
                result_frame[pre_frame_id] = box_list
                result[pre_cam_id] = result_frame
                result_frame = {}
                box_list = []
            else:
                if pre_frame_id != frame_id:
                    result_frame[pre_frame_id] = box_list
                    box_list = []
            box = (int(txt_list[3]),int(txt_list[4]),int(txt_list[5]),int(txt_list[6]))
            box_list.append(box)
            pre_frame_id = frame_id
            pre_cam_id = cam_id
    print("error ********************")
    raise


def compare_result(result_0,result_1,base_path_image,save_result_path):
    index = 0
    
    for cam_id in result_0:
        box_ = []
        for frame_id in result_0[cam_id]:
            
            if frame_id not in result_0[cam_id]:
                A = set()
            else:
                A = set(result_0[cam_id][frame_id])

            if frame_id not in result_1[cam_id]:
                B = set()
            else:
                B = set(result_1[cam_id][frame_id])
            
            diff_A = A.difference(B)
            diff_B = B.difference(A)
            if len(diff_A) == 0 and len(diff_B) == 0:
                continue

            frame_id_p = int(frame_id) - 1
            second_path = "c0"+cam_id + "/" + str(frame_id_p)+".jpg"
            image_file = os.path.join(base_path_image,second_path)
            save_image_file = os.path.join(save_result_path,second_path)
            print(image_file)
            print(save_image_file)
            image = cv2.imread(image_file)
            for box in diff_A:
                cv2.rectangle(image, (box[0], box[1]), (box[0] + box[2], box[1]+box[3]), (0, 0, 255), 2)
                cv2.putText(image, "A", (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            for box in diff_B:
                # box_.append(box)
                # if cam_id == "41":
                #     area = box[2]*box[3]
                #     if area < 700:
                #         index = index + 1
                cv2.rectangle(image, (box[0], box[1]), (box[0] + box[2], box[1]+box[3]), (255, 0, 0), 2)
                cv2.putText(image, "B", (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)
            cv2.imwrite(save_image_file,image)
        print("index:",index)
        print("box_:",len(box_))
result_a_file = "/mnt/LocalDisk1/Projects/AIC21-MTMC/reid/reid-matching/tools/track1.txt"
result_b_file = "/mnt/LocalDisk1/Projects/AIC21-MTMC/datasets/algorithm_results/xxx.txt"
base_path_image = "/mnt/LocalDisk1/Projects/AIC21-MTMC/datasets/algorithm_results/image"
save_result_path = "/mnt/LocalDisk1/Projects/AIC21-MTMC/datasets/algorithm_results/detection/image_result"
if __name__=='__main__':
    result_0 = get_result(result_a_file)
    result_1 = get_result(result_b_file)
    compare_result(result_0,result_1,base_path_image,save_result_path)

