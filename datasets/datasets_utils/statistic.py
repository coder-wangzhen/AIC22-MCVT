import cv2, random
import os 

def bb_intersection_over_union(boxA, boxB):
    boxA = [int(x) for x in boxA]
    boxB = [int(x) for x in boxB]

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

datadir = ["train", "validation", "test"]
results = ""
for i,x_path in enumerate(datadir):
    for y in os.listdir(x_path):
        if y.startswith('S'):
            y_path = os.path.join(x_path,y)
            for z in os.listdir(y_path):
                z_path = os.path.join(y_path,z)
                if z.startswith('c'):
                    results = results + z_path + ": "
                    video_path = os.path.join(z_path,'vdo.avi')
                    gt_path = os.path.join(z_path,'gt/gt.txt')
                    video = cv2.VideoCapture(video_path)
                    w, h = int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    results = results + f"{w} x {h}\n"
                    if x_path == "test":
                        continue
                    max_area = 0
                    with open(gt_path, "r") as f:
                        lines = f.readlines()
                    cur_frame = 1
                    cur_line = 0
                    area_list = []
                    iou_list = []
                    disx_list = []
                    disy_list = []
                    cur_frame_boxes = []
                    while cur_line < len(lines):
                        line = lines[cur_line].split(",")
                        if int(line[0]) == cur_frame:
                            # iou
                            cur_line_box = [float(line[2]), float(line[3]), float(line[2])+float(line[4]), float(line[3])+float(line[5])]
                            for box in cur_frame_boxes:
                                iou = bb_intersection_over_union(cur_line_box, box)
                                iou_list.append(iou)
                            cur_frame_boxes.append(cur_line_box)
                            # area
                            bw, bh = int(line[4]), int(line[5])
                            area = bw * bh
                            area_list.append(area)
                            # distance
                            centerx = float(line[2])+float(line[4])/2.0
                            centery = float(line[3])+float(line[5])/2.0
                            dis_centerx = min(centerx, w-centerx)
                            dis_centery = min(centery, h-centery)
                            disx_list.append(dis_centerx)
                            disy_list.append(dis_centery)

                            cur_line += 1
                        else:
                            cur_frame_boxes = []
                            cur_frame += 1
                    area_list.sort()
                    iou_list.sort(reverse=True)
                    disx_list.sort()
                    disy_list.sort()
                    if iou_list == []:
                        iou_list = [0]
                    results = results + f"min box area: {area_list[0]}\n"
                    results = results + f"max iou: {iou_list[0]}\n"
                    results = results + f"min distance x, y: {disx_list[0], disy_list[0]}\n\n"
    print(results)
    with open("statistic.txt", "w") as f:
        f.write(results)