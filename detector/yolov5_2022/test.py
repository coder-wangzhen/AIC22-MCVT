import torch
import cv2

from pathlib import Path
from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync


if __name__ == "__main__":
    source = "data/videos/S06C041.avi"
    source_roi = "data/videos/S06C041.jpg"
    # small_save_dir = "./runs/detect/results/small"
    large_save_dir = "./runs/detect/results/S06C041"
    txt_save_file = "./runs/detect/results/S06C041/det_yolov5x6.txt"
    device = select_device("")
    model = DetectMultiBackend("./weights/yolov5x.pt", device=device, dnn=False, data="./data/coco128.yaml")
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    imgsz = check_img_size((960,960), s=stride)  # check image size

    roi = cv2.imread(source_roi, cv2.IMREAD_GRAYSCALE)
    _, roi=cv2.threshold(roi, 127, 255, cv2.THRESH_BINARY);

    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
    bs = 1  # batch_s?ize

    dt, seen = [0.0, 0.0, 0.0], 0
    with open(txt_save_file, "w") as fw:
        for path, im, im0s, vid_cap, s, img_letter in dataset:
            im0s = cv2.bitwise_and(im0s,im0s,mask=roi)
            # cv2.imwrite("frame.jpg", im0s)
            t1 = time_sync()
            im = torch.from_numpy(im).to(device)
            im = im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            t2 = time_sync()
            dt[0] += t2 - t1

            # Inference
            pred = model(im, augment=False, visualize=False)
            t3 = time_sync()
            dt[1] += t3 - t2

            # NMS
            conf_thres=0.15
            iou_thres=0.45
            classes=(2,5,7)
            # classes=None
            agnostic_nms=True
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=1000)
            dt[2] += time_sync() - t3

            # print(pred)

            # Process predictions
            for i, det in enumerate(pred):  # per image
                seen += 1
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
                # small_save_path = f"{small_save_dir}/{frame}.jpg"     # im.jpg
                large_save_path = f"{large_save_dir}/{frame}.jpg"     # im.jpg
                s += '%gx%g ' % im.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0.copy()  # for save_crop
                # annotator = Annotator(img_letter, line_width=2, example=str(names))
                annotator0 = Annotator(im0, line_width=1, example=str(names))
                if len(det):
                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    # # Write results
                    # for *xyxy, conf, cls in reversed(det):
                    #     c = int(cls)  # integer class
                    #     label = f'{names[c]} {conf:.2f}'
                    #     annotator.box_label(xyxy, label, color=colors(c, True))

                    # Rescale boxes from img_size to im0 size
                    det0 = det.clone()
                    det0[:, :4] = scale_coords(im.shape[2:], det0[:, :4], im0.shape).round()

                    # Write results
                    for *xyxy, conf, cls in reversed(det0):
                        w, h = int(xyxy[2]-xyxy[0]), int(xyxy[3]-xyxy[1])
                        area = w*h
                        if area > 400:
                            c = int(cls)  # integer class
                            # label = f'{names[c]} {conf:.2f}'
                            label = f'{conf:.2f}'
                            annotator0.box_label(xyxy, label, color=colors(c, True))
                            result = f"{frame},-1,{xyxy[0]:.3f},{xyxy[1]:.3f},{xyxy[2]-xyxy[0]:.3f},{xyxy[3]-xyxy[1]:.3f},{conf:.3f},-1,-1,-1\n"
                            fw.write(result)
                # img_letter = annotator.result()
                img0 = annotator0.result()
                # print(save_path)
                # cv2.imwrite(small_save_path, img_letter)
                cv2.imwrite(large_save_path, img0)
            LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')
