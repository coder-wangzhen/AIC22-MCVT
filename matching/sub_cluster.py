from utils.filter import *
from utils.visual_rr import visual_rerank
from sklearn.cluster import AgglomerativeClustering
import sys, copy
sys.path.append('../')
from config import cfg

def get_sim_matrix(_cfg,cid_tid_dict,cid_tids, sub_c_to_c):
    count = len(cid_tids)
    print('count: ', count)

    q_arr = np.array([cid_tid_dict[cid_tids[i]]['mean_feat'] for i in range(count)])
    g_arr = np.array([cid_tid_dict[cid_tids[i]]['mean_feat'] for i in range(count)])
    q_arr = normalize(q_arr, axis=1)
    g_arr = normalize(g_arr, axis=1)
    # sim_matrix = np.matmul(q_arr, g_arr.T)

    # st mask
    st_mask = np.ones((count, count), dtype=np.float32)
    st_mask = intracam_ignore(st_mask, cid_tids)

    st_mask = st_filter1(st_mask, cid_tids, cid_tid_dict)

    # visual rerank
    visual_sim_matrix = visual_rerank(q_arr, g_arr, cid_tids, _cfg)
    visual_sim_matrix = visual_sim_matrix.astype('float32')
    # print(visual_sim_matrix)
    # merge result
    np.set_printoptions(precision=3)
    sim_matrix = visual_sim_matrix * st_mask

    # sim_matrix[sim_matrix < 0] = 0
    np.fill_diagonal(sim_matrix, 0)
    return sim_matrix

def normalize(nparray, axis=0):
    nparray = preprocessing.normalize(nparray, norm='l2', axis=axis)
    return nparray

def get_match(cluster_labels):
    cluster_dict = dict()
    cluster = list()
    for i, l in enumerate(cluster_labels):
        if l in list(cluster_dict.keys()):
            cluster_dict[l].append(i)
        else:
            cluster_dict[l] = [i]
    for idx in cluster_dict:
        cluster.append(cluster_dict[idx])
    return cluster

def get_cid_tid(cluster_labels,cid_tids):
    cluster = list()
    for labels in cluster_labels:
        cid_tid_list = list()
        for label in labels:
            cid_tid_list.append(cid_tids[label])
        cluster.append(cid_tid_list)
    return cluster

def combin_cluster(sub_labels,cid_tids):
    cluster = list()
    for sub_c_to_c in sub_labels:
        if len(cluster)<1:
            cluster = sub_labels[sub_c_to_c]
            continue

        for c_ts in sub_labels[sub_c_to_c]:
            is_add = False
            for i_c, c_set in enumerate(cluster):
                if len(set(c_ts) & set(c_set))>0:
                    new_list = list(set(c_ts) | set(c_set))
                    cluster[i_c] = new_list
                    is_add = True
                    break
            if not is_add:
                cluster.append(c_ts)
    labels = list()
    num_tr = 0
    for c_ts in cluster:
        label_list = list()
        for c_t in c_ts:
            label_list.append(cid_tids.index(c_t))
            num_tr+=1
        label_list.sort()
        labels.append(label_list)
    print("new tricklets:{}".format(num_tr))
    return labels,cluster

def combin_cluster2(sub_labels,cid_tids):
    sub_labels_copy = copy.deepcopy(sub_labels)
    cluster = list()
    for sub_c_to_c in sub_labels:
        if len(cluster)<1:
            cluster = sub_labels[sub_c_to_c]
            continue

        for c_ts in sub_labels[sub_c_to_c]:
            is_add = False
            for i_c, c_set in enumerate(cluster):
                if len(set(c_ts) & set(c_set))>0:
                    new_list = list(set(c_ts) | set(c_set))
                    cluster[i_c] = new_list
                    is_add = True
                    break
            if not is_add:
                cluster.append(c_ts)
    # print(cluster)

    for sub_c_to_c in sub_labels_copy:
        for c_ts in sub_labels_copy[sub_c_to_c]:
            is_add = False
            for i_c, c_set in enumerate(cluster):
                if len(set(c_ts)-set(c_set)) !=0 and len(set(c_ts) & set(c_set))>0:
                    new_list = list(set(c_ts) | set(c_set))
                    cluster[i_c] = new_list
                    is_add = True
                    break

    labels = list()
    num_tr = 0
    for c_ts in cluster:
        label_list = list()
        for c_t in c_ts:
            label_list.append(cid_tids.index(c_t))
            num_tr+=1
        label_list.sort()
        labels.append(label_list)
    print("new tricklets:{}".format(num_tr))
    return labels,cluster

def combin_feature(cid_tid_dict,sub_cluster):
    for sub_ct in sub_cluster:
        if len(sub_ct)<2: continue
        mean_feat = np.array([cid_tid_dict[i]['mean_feat'] for i in sub_ct])
        for i in sub_ct:
            cid_tid_dict[i]['mean_feat'] = mean_feat.mean(axis=0)
    return cid_tid_dict

def get_labels(_cfg, cid_tid_dict, cid_tids, score_thr):
    # 1st cluster
    sub_cid_tids = subcam_list(cid_tid_dict,cid_tids)
    # print(sub_cid_tids)
    sub_labels = dict()
    dis_thrs = [0.7,0.5,0.5,0.5,0.5,
                0.7,0.5,0.5,0.5,0.5]
    for i,sub_c_to_c in enumerate(sub_cid_tids):
        sim_matrix = get_sim_matrix(_cfg,cid_tid_dict,sub_cid_tids[sub_c_to_c], sub_c_to_c)
        cluster_labels = AgglomerativeClustering(n_clusters=None, distance_threshold=1-dis_thrs[i], affinity='precomputed',
                                linkage='complete').fit_predict(1 - sim_matrix)
        labels = get_match(cluster_labels)
        cluster_cid_tids = get_cid_tid(labels,sub_cid_tids[sub_c_to_c])
        sub_labels[sub_c_to_c] = cluster_cid_tids
    print("old tricklets:{}".format(len(cid_tids)))
    labels,sub_cluster = combin_cluster2(sub_labels,cid_tids)

    u_turn = []
    for cluster in sub_cluster:
        temp_set = set()
        for tl in cluster:
            if tl[0] not in temp_set:
                temp_set.add(tl[0])
            else:
                turn_cam = tl[0]
                u_turn.append([i for i in cluster if i[0] == turn_cam])

    # 2ed cluster
    cid_tid_dict_new = combin_feature(cid_tid_dict, sub_cluster)
    sub_cid_tids = subcam_list2(cid_tid_dict_new,cid_tids)
    sub_labels = dict()
    dis_thrs = [0.11,0.1,0.12,0.1,0.27]
    for i,sub_c_to_c in enumerate(sub_cid_tids):
        sim_matrix = get_sim_matrix(_cfg,cid_tid_dict_new,sub_cid_tids[sub_c_to_c], sub_c_to_c)

        tmp_list = list()
        a = (sim_matrix < 0.3) & (sim_matrix > 0.1)
        index = np.where(a)
        for k in range(len(index[0])):
            tmp_list.append(set([sub_cid_tids[sub_c_to_c][index[0][k]],sub_cid_tids[sub_c_to_c][index[1][k]]]))

        for tls in u_turn:
            try:
                tl1, tl2 = tls[0], tls[1]
                idx1 = sub_cid_tids[sub_c_to_c].index(tl1)
                idx2 = sub_cid_tids[sub_c_to_c].index(tl2)
                sim_matrix[idx1][idx2] = 0.5
                sim_matrix[idx2][idx1] = 0.5
            except: pass

        cluster_labels = AgglomerativeClustering(n_clusters=None, distance_threshold=1-dis_thrs[i], affinity='precomputed',
                                linkage='complete').fit_predict(1 - sim_matrix)
        labels = get_match(cluster_labels)
        cluster_cid_tids = get_cid_tid(labels,sub_cid_tids[sub_c_to_c])
        # print("cluster_cid_tids:",cluster_cid_tids)

        for kk in cluster_cid_tids:
            if len(kk) > 1:
                for j,v in enumerate(kk[:-1]):
                    if set([kk[j],kk[j+1]]) in tmp_list:
                        if abs(kk[j][1]-kk[j+1][1]) > 100:
                            cluster_cid_tids.remove(kk)

        sub_labels[sub_c_to_c] = cluster_cid_tids
    print("old tricklets:{}".format(len(cid_tids)))
    labels,sub_cluster = combin_cluster(sub_labels,cid_tids)

    # 3rd cluster
    # cid_tid_dict_new = combin_feature(cid_tid_dict,sub_cluster)
    # sim_matrix = get_sim_matrix(_cfg,cid_tid_dict_new, cid_tids)
    # cluster_labels = AgglomerativeClustering(n_clusters=None, distance_threshold=1 - 0.2, affinity='precomputed',
    #                                          linkage='complete').fit_predict(1 - sim_matrix)
    # labels = get_match(cluster_labels)
    return labels

if __name__ == '__main__':
    cfg.merge_from_file(f'../config/{sys.argv[1]}')
    cfg.freeze()
    scene_name = ['S06']
    scene_cluster = [[41, 42, 43, 44, 45, 46]]
    fea_dir = './exp/viz/test/S06/trajectory/'
    cid_tid_dict = dict()

    for pkl_path in os.listdir(fea_dir):
        cid = int(pkl_path.split('.')[0][-3:])
        with open(opj(fea_dir, pkl_path),'rb') as f:
            lines = pickle.load(f)
        for line in lines:
            tracklet = lines[line]
            tid = tracklet['tid']
            if (cid, tid) not in cid_tid_dict:
                cid_tid_dict[(cid, tid)] = tracklet

    cid_tids = sorted([key for key in cid_tid_dict.keys() if key[0] in scene_cluster[0]])
    clu = get_labels(cfg,cid_tid_dict,cid_tids,score_thr=cfg.SCORE_THR)
    print('all_clu:', len(clu))

    new_clu = list()
    for c_list in clu:
        if len(c_list) <= 1: continue
        cam_list = [cid_tids[c][0] for c in c_list]
        # if len(cam_list)!=len(set(cam_list)): 
        #     print("**************", cam_list) 
        new_clu.append([cid_tids[c] for c in c_list])
    print('new_clu: ', len(new_clu))

    all_clu = new_clu

    cid_tid_label = dict()
    for i, c_list in enumerate(all_clu):
        for c in c_list:
            cid_tid_label[c] = i + 1
    pickle.dump({'cluster': cid_tid_label}, open('test_cluster.pkl', 'wb'))
