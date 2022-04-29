
import numpy as np
import os
from tqdm import tqdm
import pandas as pd
import pickle, json
import sys
sys.path.append('../../')
from config import cfg

def load_sot_track(seq, mcmt_cfg):
    data_dir = f'{mcmt_cfg.DATA_DIR}/{seq}/result'
    results = []
    for f in os.listdir(data_dir):
        track_file = os.path.join(data_dir, f)
        track_info = json.load(open(track_file,"r"))

        cur_frame = track_info["start_frame_id"]
        cur_feature = 0
        cur_box = 0
        for i,v in enumerate(track_info["is_box"]):
            if v == 1:
                if track_info["is_feature"][i] == 1:
                    frame = cur_frame
                    pid = track_info["track_id"]
                    bbox = np.array(track_info['box_list'][cur_box]).astype('float32')
                    feat = np.array(track_info['feature_list'][cur_feature])
                    dummpy_input = np.array([frame, pid, bbox[0], bbox[1], bbox[2], bbox[3], \
                                            track_info["start_region_id"], track_info["end_region_id"], \
                                            track_info["start_frame_id"], track_info["end_frame_id"]])
                    dummpy_input = np.concatenate((dummpy_input, feat))
                    results.append(dummpy_input)
                    cur_feature += 1
                    cur_box += 1
                else:
                    cur_box += 1
            cur_frame += 1
    ret = sorted(results,key=lambda x:(x[0],x[1]))
    return np.array(ret)

def eval_seq(seq, pp='', split='train', mcmt_cfg=None):
    trk_file = f'{mcmt_cfg.DATA_DIR}/{seq}/{seq}_mot.txt'
    print('loading tracked file ' + trk_file)
    results = load_sot_track(seq, mcmt_cfg)
    # Store results.
    save_pickle(results, seq, mcmt_cfg)
    trk_dir = os.path.dirname(trk_file)
    if not os.path.exists(os.path.join(trk_dir, 'res')):
        os.makedirs(os.path.join(trk_dir, 'res'))
    output_file = os.path.join(trk_dir, 'res', os.path.basename(trk_file))
    f = open(output_file, 'w')
    for row in results:
        print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (
            row[0], row[1], row[2], row[3], row[4], row[5]),file=f)
    f.close()

def save_pickle(results, sequence_name, mcmt_cfg):
    """Save pickle."""

    feat_pkl_file = f'{mcmt_cfg.DATA_DIR}/{sequence_name}/{sequence_name}_mot_feat.pkl'
    mot_feat_dic = {}
    for row in results:
        [fid, pid, x, y, w, h] = row[:6]    # pylint: disable=invalid-name
        fid = int(fid)
        pid = int(pid)
        feat = np.array(row[-2048:])
        image_name = f'{sequence_name}_{pid}_{fid}.png'
        bbox = (x, y, x+w, y+h)
        frame = f'img{int(fid):06d}'
        start_region_id = int(row[6])
        end_region_id = int(row[7])
        start_frame_id = int(row[8])
        end_frame_id = int(row[9])
        mot_feat_dic[image_name] = {'bbox': bbox, 'frame': frame, 'id': pid,
                                    'imgname': image_name, 'start_region_id':start_region_id, 
                                    'end_region_id':end_region_id, 'start_frame_id':start_frame_id, 
                                    'end_frame_id':end_frame_id, 'feat': feat}
    pickle.dump(mot_feat_dic, open(feat_pkl_file, 'wb'), pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    import sys
    print('sys args ars: ', sys.argv)
    cfg.merge_from_file(f'../../config/{sys.argv[3]}')
    cfg.freeze()
    eval_seq(sys.argv[1], pp=sys.argv[2], mcmt_cfg=cfg)
