from xml.etree.ElementPath import get_parent_map
import cv2, random
import os 

if __name__ == "__main__":
    # detection ---------------------------------------------------------------------------------------------
    
    # video_path = "../AIC22_Track1_MTMC_Tracking/train/S01/c001/vdo.avi"
    # # det_result_path = "../AIC22_Track1_MTMC_Tracking/train/S01/c001/det/det_mask_rcnn.txt"
    # det_result_path = "../AIC22_Track1_MTMC_Tracking/train/S01/c001/det/det_yolo3.txt"
    # # det_result_path = "../AIC22_Track1_MTMC_Tracking/train/S01/c001/det/det_ssd512.txt"
    # # save_path = "../AIC22_Track1_MTMC_Tracking/train/S01/c001/det/det_mask_rcnn"
    # save_path = "../AIC22_Track1_MTMC_Tracking/train/S01/c001/det/det_yolo3"
    # # save_path = "../AIC22_Track1_MTMC_Tracking/train/S01/c001/det/det_ssd512"
    # cap = cv2.VideoCapture(video_path)
    # num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # print("frames:", num_frames)

    # cur_frame = 1
    # cur_line = 0

    # with open(det_result_path,"r") as f:
    #     lines = f.readlines()    
    #     # print("lines[0]:",lines[0].split(","))
    
    # while(True):
    #     ret, frame = cap.read()
    #     if ret:
    #         # cv2.imshow("frame", frame)
    #         # if cv2.waitKey(1) & 0xFF == ord('q'):
    #         #     break
    #         while cur_line < len(lines):
    #             line = lines[cur_line].split(",")
    #             if int(line[0]) == cur_frame:
    #                 # draw
    #                 p1, p2 = (round(float(line[2])), round(float(line[3]))), (round(float(line[2])+float(line[4])), round(float(line[3])+float(line[5])))
    #                 # print(p1,p2)
    #                 cv2.rectangle(frame, p1, p2, (0, 0, 255), thickness=3, lineType=cv2.LINE_AA)
    #                 w, h = cv2.getTextSize(line[6], 0, fontScale=2 / 3, thickness=1)[0]  # text width, height
    #                 outside = p1[1] - h - 3 >= 0  # label fits outside box
    #                 p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
    #                 cv2.rectangle(frame, p1, p2, (0, 0, 255), -1, cv2.LINE_AA)  # filled
    #                 cv2.putText(frame, line[6], (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), 0, 2 / 3, (255, 255, 255),
    #                             thickness=1, lineType=cv2.LINE_AA)
    #                 cur_line += 1
    #             else:
    #                 break
    #         cv2.putText(frame, str(cur_frame), (50,50), 0, 2 / 3, (255, 0, 0), thickness=1, lineType=cv2.LINE_AA)
    #         # cv2.imshow("frame", frame)
    #         # if cv2.waitKey(1) & 0xFF == ord('q'):
    #         #     break
    #         cv2.imwrite(f"{save_path}/{cur_frame}.jpg", frame)
    #         cur_frame += 1
    #     else:
    #         break
    # cap.release()

    # gt -------------------------------------------------------------------------------------

    # # video_path = "../AIC22_Track1_MTMC_Tracking/train/S01/c002/vdo.avi"
    # video_path = "../AIC22_Track1_MTMC_Tracking/train/S01/c005/vdo.avi"
    # # det_result_path = "../AIC22_Track1_MTMC_Tracking/train/S01/c005/mtsc/mtsc_tc_yolo3.txt" # mtsc_deepsort_mask_rcnn
    # det_result_path = "../AIC22_Track1_MTMC_Tracking/train/S01/c005/gt/gt.txt"
    # # save_path = "../AIC22_Track1_MTMC_Tracking/train/S01/c005/mtsc/mtsc_tc_yolo3" # mtsc_deepsort_mask_rcnn
    # save_path = "../AIC22_Track1_MTMC_Tracking/train/S01/c005/gt/gt"
    # cap = cv2.VideoCapture(video_path)
    # num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # print("frames:", num_frames)
 
    # colors = []
    # random.seed(100)
    # for i in range(100):
    #     color = (random.randint(0,100),random.randint(0,100),random.randint(0,100))
    #     colors.append(color)
    # # print(colors)
 
    # cur_frame = 1
    # cur_line = 0
 
    # with open(det_result_path,"r") as f:
    #     lines = f.readlines()    
    #     # print("lines[0]:",lines[0].split(","))
    
    # area_list = []
    # while(True):
    #     ret, frame = cap.read()
    #     if ret:
    #         # cv2.imshow("frame", frame)
    #         # if cv2.waitKey(1) & 0xFF == ord('q'):
    #         #     break
    #         while cur_line < len(lines):
    #             line = lines[cur_line].split(",")
    #             if int(line[0]) == cur_frame:
    #                 # draw
    #                 p1, p2 = (round(float(line[2])), round(float(line[3]))), (round(float(line[2])+float(line[4])), round(float(line[3])+float(line[5])))
    #                 # print(p1,p2)
    #                 w, h = p2[0]-p1[0], p2[1]-p1[1]
    #                 area = w *h
    #                 area_list.append(area)
 
    #                 track_id = int(line[1])
    #                 cv2.rectangle(frame, p1, p2, colors[track_id%100], thickness=3, lineType=cv2.LINE_AA)
    #                 w, h = cv2.getTextSize(line[1], 0, fontScale=2 / 3, thickness=1)[0]  # text width, height
    #                 outside = p1[1] - h - 3 >= 0  # label fits outside box
    #                 p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
    #                 cv2.rectangle(frame, p1, p2, colors[track_id%100], -1, cv2.LINE_AA)  # filled
    #                 cv2.putText(frame, line[1], (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), 0, 2 / 3, (255, 255, 255),
    #                             thickness=1, lineType=cv2.LINE_AA)
    #                 cur_line += 1
    #             else:
    #                 break
    #         # cv2.imshow("frame", frame)
    #         # if cv2.waitKey(1) & 0xFF == ord('q'):
    #         #     break
    #         cv2.imwrite(f"{save_path}/{cur_frame}.jpg", frame)
    #         cur_frame += 1
    #     else:
    #         break
    
    # area_list.sort()
    # with open("./train/S01/c005/gt/gt_area.txt", "w") as f:
    #     for item in area_list:
    #         f.write(str(item))
    #         f.write("\n")
    # cap.release()

    # track results -------------------------------------------------------------------------------------

    test_path = "../AIC22_Track1_MTMC_Tracking/test/S06"
    save_path = "../AIC22_Track1_MTMC_Tracking/algorithm_results/final_result/result"
    track_result_path = "../../reid/reid-matching/tools/track1.txt"
    cams = ("c041","c042","c043","c044","c045","c046")
    # cams = ("c044","c045","c046")
    # save_path = "../../reid/reid-matching/tools/tmp_result_before"
    # track_result_path = "../../reid/reid-matching/tools/tmp_result_before/41.txt"
    # cams = ("c041",)

    colors = []
    random.seed(100)
    for i in range(100):
        color = (random.randint(0,100),random.randint(0,100),random.randint(0,100))
        colors.append(color)
    # print(colors)

    with open(track_result_path,"r") as f:
        data = f.readlines()
        # print("data[0]:",data[0].split(","))

    for cam in cams:
        print(f"processing {cam} ...")
        if not os.path.isdir(f"{save_path}/{cam}"):
            os.makedirs(f"{save_path}/{cam}")

        cap = cv2.VideoCapture(os.path.join(test_path,cam,"vdo.avi"))
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # print("frames:", num_frames)

        lines = [line.split(" ") for line in data if line[0:2] == cam[2:]]
        lines.sort(key = lambda x:int(x[2]), reverse=False)
        # print(len(lines))

        cur_frame = 1
        cur_line = 0
        
        while(True):
            ret, frame = cap.read()
            if ret:
                # cv2.imshow("frame", frame)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break
                while cur_line < len(lines):
                    # print(cur_line)
                    line = lines[cur_line]
                    if int(line[2]) == cur_frame:
                        # draw
                        p1, p2 = (round(float(line[3])), round(float(line[4]))), (round(float(line[3])+float(line[5])), round(float(line[4])+float(line[6])))
                        # print(p1,p2)
                        w, h = p2[0]-p1[0], p2[1]-p1[1]
    
                        track_id = int(line[1])
                        cv2.rectangle(frame, p1, p2, colors[track_id%100], thickness=3, lineType=cv2.LINE_AA)
                        w, h = cv2.getTextSize(line[1], 0, fontScale=2 / 3, thickness=1)[0]  # text width, height
                        outside = p1[1] - h - 3 >= 0  # label fits outside box
                        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
                        cv2.rectangle(frame, p1, p2, colors[track_id%100], -1, cv2.LINE_AA)  # filled
                        cv2.putText(frame, line[1], (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), 0, 2 / 3, (255, 255, 255),
                                    thickness=1, lineType=cv2.LINE_AA)
                        cur_line += 1
                    else:
                        break
                # cv2.imshow("frame", frame)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break
                cv2.imwrite(f"{save_path}/{cam}/{cur_frame}.jpg", frame)
                cur_frame += 1
            else:
                break
        cap.release()
