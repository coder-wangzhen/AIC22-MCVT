import os
from os.path import join as opj
import scipy.io as scio
import cv2
import numpy as np
import pickle
from scipy import spatial
import copy
import multiprocessing
from math import *
from sklearn import preprocessing
import tqdm

CAM_DIST = [[  0, 40, 55,100,120,145],
            [ 40,  0, 15, 60, 80,105],
            [ 55, 15,  0, 40, 65, 90],
            [100, 60, 40,  0, 20, 45],
            [120, 80, 65, 20,  0, 25],
            [145,105, 90, 45, 25,  0]]

SPEED_LIMIT = [[(0,0), (400,1300), (550,2000), (1000,2000), (1200, 2000), (1450, 2000)],
               [(400,1300), (0,0), (100,900), (600,2000), (800,2000), (1050,2000)],
               [(550,2000), (100,900), (0,0), (350,1050), (650,2000), (900, 2000)],
               [(1000,2000), (600,2000), (350,1050), (0,0), (150,500), (450, 2000)],
               [(1200, 2000), (800,2000), (650,2000), (150,500), (0,0), (250,900)],
               [(1450, 2000), (1050,2000), (900, 2000), (450, 2000), (250,900), (0,0)]]  

rotate_270 = lambda i: [3,4,5,6,7,8,1,2,10][i % 10 - 1] # 所有区域顺时针旋转270度,中心区域不变

def intracam_ignore(st_mask, cid_tids):
    count = len(cid_tids)
    for i in range(count):
        for j in range(count):
            if cid_tids[i][0] == cid_tids[j][0]:
                st_mask[i, j] = 0.
    return st_mask

def get_dire(zone_list,cid):
    zs,ze = zone_list[0],zone_list[-1]
    return (zs,ze)

def st_filter1(st_mask, cid_tids,cid_tid_dict):
    count = len(cid_tids)
    # west_in = {1,10}
    # west_out = {2,10}
    # east_in = {5,10}
    # east_out = {6,10}
    west = {3,0}
    east = {4,0}
    for i in range(count):
        i_tracklet = cid_tid_dict[cid_tids[i]]
        i_cid = i_tracklet['cam']
        i_start = i_tracklet['zone_list'][0]
        i_end = i_tracklet['zone_list'][-1]
        i_frame = [int(k * 10) for k in i_tracklet['io_time']]
        for j in range(count):
            j_tracklet = cid_tid_dict[cid_tids[j]]
            j_cid = j_tracklet['cam']
            j_start = j_tracklet['zone_list'][0]
            j_end = j_tracklet['zone_list'][-1]
            j_frame = [int(k * 10) for k in j_tracklet['io_time']]
            frame_range = SPEED_LIMIT[i_cid-41][j_cid-41]
            forward_range = range(i_frame[1] + frame_range[0], i_frame[1] + frame_range[1])
            reverse_range = range(i_frame[0] - frame_range[1], i_frame[0] - frame_range[0])
            
            match = False
            if i_cid < j_cid:
                if i_end in west and j_start in east and j_frame[0] in forward_range: # 向西
                    match = True
                if i_start in west and j_end in east and j_frame[1] in reverse_range: # 向东倒放
                    match = True
            if i_cid > j_cid:
                if i_start in east and j_end in west and j_frame[1] in reverse_range: # 向西倒放
                    match = True
                if i_end in east and j_start in west and j_frame[0] in forward_range: # 向东
                    match = True
                    
            if not match:
                st_mask[i, j] = 0.0
                st_mask[j, i] = 0.0
    return st_mask

def st_filter(st_mask, cid_tids,cid_tid_dict):
    count = len(cid_tids)
    for i in range(count):
        i_tracklet = cid_tid_dict[cid_tids[i]]
        i_cid = i_tracklet['cam']
        i_dire = get_dire(i_tracklet['zone_list'],i_cid)
        i_iot = i_tracklet['io_time']
        for j in range(count):
            j_tracklet = cid_tid_dict[cid_tids[j]]
            j_cid = j_tracklet['cam']
            j_dire = get_dire(j_tracklet['zone_list'], j_cid)
            j_iot = j_tracklet['io_time']

            match_dire = True
            cam_dist = CAM_DIST[i_cid-41][j_cid-41]
            # 如果时间重叠
            if i_iot[0]-cam_dist<j_iot[0] and j_iot[0]<i_iot[1]+cam_dist:
                match_dire = False
            if i_iot[0]-cam_dist<j_iot[1] and j_iot[1]<i_iot[1]+cam_dist:
                match_dire = False

            # 出去后不再匹配
            if i_dire[1] in [1,2]: # i out
                if i_iot[0] < j_iot[1]+cam_dist:
                    match_dire = False

            if i_dire[1] in [1,2]:
                if i_dire[0] in [3] and i_cid>j_cid:
                    match_dire = False
                if i_dire[0] in [4] and i_cid<j_cid:
                    match_dire = False

            if i_cid in [41] and i_dire[1] in [4]:
                if i_iot[0] < j_iot[1]+cam_dist:
                    match_dire = False
                if i_iot[1] > 199:
                    match_dire = False
            if i_cid in [46] and i_dire[1] in [3]:
                if i_iot[0] < j_iot[1]+cam_dist:
                    match_dire = False

            # 进入才匹配
            if i_dire[0] in [1,2]:  # i in
                if i_iot[1] > j_iot[0]-cam_dist:
                    match_dire = False

            if i_dire[0] in [1,2]:
                if i_dire[1] in [3] and i_cid>j_cid:
                    match_dire = False
                if i_dire[1] in [4] and i_cid<j_cid:
                    match_dire = False

            is_ignore = False
            if ((i_dire[0] == i_dire[1] and i_dire[0] in [3, 4]) or (
                    j_dire[0] == j_dire[1] and j_dire[0] in [3, 4])):
                is_ignore = True

            if not is_ignore:
                # 方向冲突
                if (i_dire[0] in [3] and j_dire[0] in [4]) or (i_dire[1] in [3] and j_dire[1] in [4]):
                    match_dire = False

                # 去下一场景之前的过滤
                if i_dire[1] in [3] and i_cid < j_cid:
                    if i_iot[1]>j_iot[1]-cam_dist:
                        match_dire = False
                if i_dire[1] in [4] and i_cid > j_cid:
                    if i_iot[1]>j_iot[1]-cam_dist:
                        match_dire = False

                if i_dire[0] in [3] and i_cid < j_cid:
                    if i_iot[0]<j_iot[0]+cam_dist:
                        match_dire = False
                if i_dire[0] in [4] and i_cid > j_cid:
                    if i_iot[0]<j_iot[0]+cam_dist:
                        match_dire = False
                ## ↑ 3-30

                ## 4-1
                if i_dire[0] in [3] and i_cid > j_cid:
                    if i_iot[1]>j_iot[0]-cam_dist:
                        match_dire = False
                if i_dire[0] in [4] and i_cid < j_cid:
                    if i_iot[1]>j_iot[0]-cam_dist:
                        match_dire = False
                ##

                # 去下一场景之后的过滤
                ## 4-7
                if i_dire[1] in [3] and i_cid > j_cid:
                    if i_iot[0]<j_iot[1]+cam_dist:
                        match_dire = False
                if i_dire[1] in [4] and i_cid < j_cid:
                    if i_iot[0]<j_iot[1]+cam_dist:
                        match_dire = False
                ##
            else:
                if i_iot[1]>199:
                    if i_dire[0] in [3] and i_cid < j_cid:
                        if i_iot[0] < j_iot[0] + cam_dist:
                            match_dire = False
                    if i_dire[0] in [4] and i_cid > j_cid:
                        if i_iot[0] < j_iot[0] + cam_dist:
                            match_dire = False
                    if i_dire[0] in [3] and i_cid > j_cid:
                        match_dire = False
                    if i_dire[0] in [4] and i_cid < j_cid:
                        match_dire = False
                if i_iot[0]<1:
                    if i_dire[1] in [3] and i_cid > j_cid:
                            match_dire = False
                    if i_dire[1] in [4] and i_cid < j_cid:
                            match_dire = False

            if not match_dire:
                st_mask[i, j] = 0.0
                st_mask[j, i] = 0.0
    return st_mask

def subcam_list(cid_tid_dict,cid_tids):
    sub_3_4 = dict()
    sub_4_3 = dict()
    for cid_tid in cid_tids:
        cid,tid = cid_tid
        tracklet = cid_tid_dict[cid_tid]
        zs,ze = get_dire(tracklet['zone_list'], cid)
        if zs in [3] and cid not in [46]: # 4 to 3
            if not cid+1 in sub_4_3:
                sub_4_3[cid+1] = []
            sub_4_3[cid + 1].append(cid_tid)
        if ze in [4] and cid not in [41]: # 4 to 3
            if not cid in sub_4_3:
                sub_4_3[cid] = []
            sub_4_3[cid].append(cid_tid)
        if zs in [4] and cid not in [41]: # 3 to 4
            if not cid-1 in sub_3_4:
                sub_3_4[cid-1] = []
            sub_3_4[cid - 1].append(cid_tid)
        if ze in [3] and cid not in [46]: # 3 to 4
            if not cid in sub_3_4:
                sub_3_4[cid] = []
            sub_3_4[cid].append(cid_tid)
    sub_cid_tids = dict()
    for i in sub_3_4:
        sub_cid_tids[(i,i+1)]=sub_3_4[i]
    for i in sub_4_3:
        sub_cid_tids[(i,i-1)]=sub_4_3[i]
    return sub_cid_tids

def subcam_list2(cid_tid_dict,cid_tids):
    sub_dict = dict()
    for cid_tid in cid_tids:
        cid, tid = cid_tid
        if cid not in [41]:
            if not cid in sub_dict:
                sub_dict[cid] = []
            sub_dict[cid].append(cid_tid)
        if cid not in [46]:
            if not cid+1 in sub_dict:
                sub_dict[cid+1] = []
            sub_dict[cid+1].append(cid_tid)
    return sub_dict