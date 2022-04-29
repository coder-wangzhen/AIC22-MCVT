#include "linear_assignment.h"
#include "hungarianoper.h"
#include <map>

linear_assignment *linear_assignment::instance = NULL;
linear_assignment::linear_assignment(tracker* ptr_t_p)
{
    ptr_t = ptr_t_p;
}

// linear_assignment *linear_assignment::getInstance()
// {
//     if(instance == NULL) instance = new linear_assignment();
//     return instance;
// }

TRACHER_MATCHD
linear_assignment::matching_cascade(
        tracker *distance_metric,
        tracker::GATED_METRIC_FUNC distance_metric_func_1,
        tracker::GATED_METRIC_FUNC distance_metric_func_2,
        tracker::GATED_METRIC_FUNC distance_metric_func_3,
        float max_distance,
        int cascade_depth,
        std::vector<Track> &tracks,
        const DETECTIONS &detections,
        std::vector<int>& track_indices,
        std::vector<int>& detection_indices,cv::Mat &frame,float merger_distance,bool is_cascade)
{
    //std::cout<<"start to cascade match"<<std::endl;
    TRACHER_MATCHD res;
    if(!is_cascade)
    {
        for(size_t i = 0; i < detections.size(); i++) 
        {
            detection_indices.push_back(int(i));
        }
    }

    std::vector<int> unmatched_detections;
    unmatched_detections.assign(detection_indices.begin(), detection_indices.end());
    res.matches.clear();
    std::vector<int> track_indices_l;
    std::map<int, int> matches_trackid;
    int level = 0;
    if(is_cascade)
    {
        level = 1;
    }
    for(; level < cascade_depth; level++) 
    {
        if(unmatched_detections.size() == 0) 
        {
            break; //No detections left;
        }

        track_indices_l.clear();
        for(int k:track_indices) 
        {
            if(tracks[k].time_since_update == 1+level)
            {
                track_indices_l.push_back(k);
            }
                
        }
        
        if(track_indices_l.size() == 0) 
        {
            continue; //Nothing to match at this level.
        }

        TRACHER_MATCHD tmp;
        if(level == 0)   //
        {
            tmp = min_cost_matching(
                    distance_metric, distance_metric_func_1,distance_metric_func_2,distance_metric_func_3,
                    max_distance, tracks, detections, track_indices_l,
                    unmatched_detections,false);
        }
        else
        {
            tmp = min_cost_matching(
                    distance_metric, distance_metric_func_1,distance_metric_func_2,distance_metric_func_3,
                    merger_distance, tracks, detections, track_indices_l,
                    unmatched_detections,false,true);
        }   
        

        unmatched_detections.assign(tmp.unmatched_detections.begin(), tmp.unmatched_detections.end());

        for(size_t i = 0; i < tmp.matches.size(); i++) 
        {
            MATCH_DATA pa = tmp.matches[i];
            int track_idx = pa.first;
            int detection_idx = pa.second;
            cv::Rect track_box = tracks[track_idx].track_box;
            DETECTION_ROW detection_t = detections[detection_idx];
            DETECTBOX tmpbox = detection_t.tlwh;
            cv::Rect detect_rect(tmpbox(0), tmpbox(1), tmpbox(2), tmpbox(3));
            if(level == 0)
            {
                bool ret = ptr_t->is_avaliable_match(track_idx,track_box,detect_rect,false);
                if(ret)
                {
                    res.matches.push_back(pa);
                    matches_trackid.insert(pa);
                }
                else
                {
                    unmatched_detections.push_back(detection_idx);
                }
            }
            else
            {
                bool ret = ptr_t->is_avaliable_match(track_idx,track_box,detect_rect,true);
                if(ret)
                {
                    res.matches.push_back(pa);
                    matches_trackid.insert(pa);
                }
                else
                {
                    unmatched_detections.push_back(detection_idx);
                }
            }  
        }
    }

    for(MATCH_DATA& data:res.matches) 
    {
        int track_idx = data.first;
        int detection_idx = data.second;
        DETECTBOX tmpbox = detections[detection_idx].tlwh;
        cv::Rect2f current_rect(tmpbox(0), tmpbox(1), tmpbox(2), tmpbox(3));
        cv::Rect current_box = cv::Rect(tmpbox(0), tmpbox(1), tmpbox(2), tmpbox(3));
        tracks[track_idx].pre_rect = current_rect;
        tracks[track_idx].status = CASCADE;
        tracks[track_idx].track_box = current_box;
        double iou_score = bbox_iou(tracks[track_idx],detections[detection_idx]);
        if(iou_score < 0.5)
        {
            tracks[track_idx].init_eco_track(frame);
        }
    }

    res.unmatched_detections.assign(unmatched_detections.begin(), unmatched_detections.end());
    for(size_t i = 0; i < track_indices.size(); i++) 
    {
        int tid = track_indices[i];
        if(matches_trackid.find(tid) == matches_trackid.end())
        {
            res.unmatched_tracks.push_back(tid);
        } 
    }
    return res;
}

TRACHER_MATCHD
linear_assignment::min_cost_matching(tracker *distance_metric,
        tracker::GATED_METRIC_FUNC distance_metric_func_1,
        tracker::GATED_METRIC_FUNC distance_metric_func_2,
        tracker::GATED_METRIC_FUNC distance_metric_func_3,
        float max_distance,
        std::vector<Track> &tracks,
        const DETECTIONS &detections,
        std::vector<int> &track_indices,
        std::vector<int> &detection_indices,bool is_iou,bool is_match_miss)
{
    TRACHER_MATCHD res;
    
    if((detection_indices.size() == 0) || (track_indices.size() == 0)) 
    {
        res.matches.clear();
        res.unmatched_tracks.assign(track_indices.begin(), track_indices.end());
        res.unmatched_detections.assign(detection_indices.begin(), detection_indices.end());
        return res;
    }
    DYNAMICM cost_matrix_1 = (distance_metric->*(distance_metric_func_1))(
                tracks, detections, track_indices, detection_indices);
    
    // std::cout<<"cost_matrix_1: "<<std::endl;
    // std::cout<<cost_matrix_1<<std::endl;
    // std::cout<<"\r\n"<<std::endl;
    if(is_match_miss)
    {
        for(int i = 0; i < cost_matrix_1.rows(); i++) {
            for(int j = 0; j < cost_matrix_1.cols(); j++) {
                float tmp = cost_matrix_1(i,j);
                if(tmp > max_distance)
                {
                    cost_matrix_1(i,j) = max_distance + 1e-5;
                } 
            }
        }
        Eigen::Matrix<float, -1, 2, Eigen::RowMajor> indices = HungarianOper::Solve(cost_matrix_1);
        res.matches.clear();
        res.unmatched_tracks.clear();
        res.unmatched_detections.clear();
        for(size_t col = 0; col < detection_indices.size(); col++) 
        {
            bool flag = false;
            for(int i = 0; i < indices.rows(); i++)
            {
                if(indices(i, 1) == col) 
                { 
                    flag = true; 
                    break;
                }
            }
                
            if(flag == false)
            {
                res.unmatched_detections.push_back(detection_indices[col]);
            }
        }
        for(size_t row = 0; row < track_indices.size(); row++) 
        {
            bool flag = false;
            for(int i = 0; i < indices.rows(); i++)
            {
                if(indices(i, 0) == row) 
                { 
                    flag = true; break; 
                }
            }
                
            if(flag == false)
            {
                res.unmatched_tracks.push_back(track_indices[row]);
            } 
        }
        for(int i = 0; i < indices.rows(); i++) 
        {
            int row = indices(i, 0);
            int col = indices(i, 1);

            int track_idx = track_indices[row];
            int detection_idx = detection_indices[col];
            if(cost_matrix_1(row, col) > max_distance) 
            {
                res.unmatched_tracks.push_back(track_idx);
                res.unmatched_detections.push_back(detection_idx);
            } 
            else
            {
                res.matches.push_back(std::make_pair(track_idx, detection_idx));
            } 
            
        }
        return res;
    }
    
    if(is_iou)
    {
        for(int i = 0; i < cost_matrix_1.rows(); i++)
        {
            for(int j = 0; j < cost_matrix_1.cols(); j++) 
            {
                float tmp = cost_matrix_1(i,j);
                if(tmp  == 1.0)
                {
                    continue;
                }
                else
                {
                    cost_matrix_1(i,j) = cost_matrix_1(i,j)*1.2;
                }
                
                // if(tmp > max_distance)
                // {
                //     cost_matrix(i,j) =  (1e-5+max_distance);
                // } 
            }
        }
    }
    

    DYNAMICM cost_matrix_2 = (distance_metric->*(distance_metric_func_2))(
                tracks, detections, track_indices, detection_indices);
    
    // std::cout<<"cost_matrix_2: "<<std::endl;
    // std::cout<<cost_matrix_2<<std::endl;
    // std::cout<<"\r\n"<<std::endl;
    DYNAMICM cost_matrix_3 = (distance_metric->*(distance_metric_func_3))(
                tracks, detections, track_indices, detection_indices);
    
    // std::cout<<"cost_matrix_3: "<<std::endl;
    // std::cout<<cost_matrix_3<<std::endl;
    // std::cout<<"\r\n"<<std::endl;

    DYNAMICM cost_matrix = cost_matrix_1 + cost_matrix_2 + cost_matrix_3;

    // std::cout<<"cost_matrix: "<<std::endl;
    // std::cout<<cost_matrix<<std::endl;
    // std::cout<<"\r\n"<<std::endl;

    for(int i = 0; i < cost_matrix.rows(); i++)
    {
        for(int j = 0; j < cost_matrix.cols(); j++) 
        {
            float tmp = cost_matrix(i,j);
            if(tmp > max_distance)
            {
                cost_matrix(i,j) =  (1e-5+max_distance);
            } 
        }
    }
    // std::cout<<"final cost_matrix: "<<std::endl;
    // std::cout<<cost_matrix<<std::endl;
    // std::cout<<"\r\n"<<std::endl;
    Eigen::Matrix<float, -1, 2, Eigen::RowMajor> indices = HungarianOper::Solve(cost_matrix);
    res.matches.clear();
    res.unmatched_tracks.clear();
    res.unmatched_detections.clear();
    for(size_t col = 0; col < detection_indices.size(); col++) 
    {
        bool flag = false;
        for(int i = 0; i < indices.rows(); i++)
        {
            if(indices(i, 1) == col) 
            { 
                flag = true; 
                break;
            }
        }
            
        if(flag == false)
        {
            res.unmatched_detections.push_back(detection_indices[col]);
        } 
    }
    for(size_t row = 0; row < track_indices.size(); row++) 
    {
        bool flag = false;
        for(int i = 0; i < indices.rows(); i++)
        {
            if(indices(i, 0) == row) 
            {
                 flag = true; 
                 break; 
            }
        }
            
        if(flag == false) 
        {
            res.unmatched_tracks.push_back(track_indices[row]);
        }    
    }
    for(int i = 0; i < indices.rows(); i++) 
    {
        int row = indices(i, 0);
        int col = indices(i, 1);

        int track_idx = track_indices[row];
        int detection_idx = detection_indices[col];
        if(cost_matrix(row, col) > max_distance) 
        {
            res.unmatched_tracks.push_back(track_idx);
            res.unmatched_detections.push_back(detection_idx);
        } 
        else 
        {
            res.matches.push_back(std::make_pair(track_idx, detection_idx));
        }
    }
    return res;
}

DYNAMICM
linear_assignment::gate_cost_matrix(
        KalmanFilterTrack *kf,
        DYNAMICM &cost_matrix,
        std::vector<Track> &tracks,
        const DETECTIONS &detections,
        const std::vector<int> &track_indices,
        const std::vector<int> &detection_indices,
        float gated_cost, bool only_position)
{
    int gating_dim = (only_position == true?2:4);
    double gating_threshold = KalmanFilterTrack::chi2inv95[gating_dim];
    std::vector<DETECTBOX> measurements;
    for(int i:detection_indices) {
        DETECTION_ROW t = detections[i];
        measurements.push_back(t.to_xyah());
    }
    for(size_t i  = 0; i < track_indices.size(); i++) {
        Track& track = tracks[track_indices[i]];
        Eigen::Matrix<float, 1, -1> gating_distance = kf->gating_distance(
                    track.mean, track.covariance, measurements, only_position);
        for (int j = 0; j < gating_distance.cols(); j++) 
        {
            //cost_matrix(i, j) = 1;
            if (gating_distance(0, j) > gating_threshold)
            {
                cost_matrix(i, j) = gated_cost;
            }  
        }
    }
    return cost_matrix;
}


double linear_assignment::bbox_iou(Track& track,const DETECTION_ROW& det)
{
    cv::Rect track_bbox = track.eco_rect;
    DETECTBOX tmpbox = det.tlwh;
    cv::Rect detect_bbox(tmpbox(0), tmpbox(1), tmpbox(2), tmpbox(3));
    cv::Rect rect_union = track_bbox | detect_bbox;

    if(rect_union.area() == 0)
    {
        std::cout<<"WARNING:rect_union.area()==0"<<std::endl;
        return 0.0;
    }

    cv::Rect intersetion = track_bbox & detect_bbox;

    if(intersetion.area() == 0)
    {
        //std::cout<<"WARNING:intersetion.area() = 0;"<<std::endl;
        return 0.0;
    }

    double IOU = intersetion.area() *1.0/ rect_union.area()*1.0;

    return IOU;
    
}

