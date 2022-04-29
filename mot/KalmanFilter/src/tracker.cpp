#include "tracker.h"
#include "nn_matching.h"
#include "model.h"
#include "linear_assignment.h"
using namespace std;

//#define MY_inner_DEBUG
#ifdef MY_inner_DEBUG
#include <string>
#include <iostream>
#endif

tracker::tracker(/*NearNeighborDisMetric *metric,*/CityTracker * city_track_ptr,
		float max_cosine_distance,float merger_distance, int nn_budget,
		float max_iou_distance, int max_age, int n_init)
{
    this->metric = new NearNeighborDisMetric(
    		NearNeighborDisMetric::METRIC_TYPE::cosine,
    		max_cosine_distance, nn_budget);

    
    this->m_merger_distance = merger_distance;
    this->max_iou_distance = max_iou_distance;
    this->max_age = max_age;
    this->n_init = n_init;

    this->kf = new KalmanFilterTrack();
    this->tracks.clear();
    this->_next_idx = 1;
    this->m_city_track_ptr = city_track_ptr;
    current_frame_id = -1;
    save_path = "../../datasets/algorithm_results/detect_merge/";

    this->linear_assignment_ptr = std::make_shared<linear_assignment>(this);
    is_running_ = true;
    run_thread_ = std::thread(std::bind(&tracker::WriteRun, this));
}
tracker::~tracker()
{
    is_running_= false;
    if (run_thread_.joinable())
    {
        run_thread_.join();
    }
}

bool tracker::WriteToLocal(std::shared_ptr<TRACK_INFO> track_info_ptr)
{

    int have_box = 1;
    int not_box = 0;
    int have_feature = 1;
    int not_feature = 0;
    rapidjson::Document jsonDoc;    //生成一个dom元素Document
    rapidjson::Document::AllocatorType &allocator = jsonDoc.GetAllocator(); //获取分配器
    jsonDoc.SetObject();

    jsonDoc.AddMember("track_id", track_info_ptr->track_id, allocator);
    jsonDoc.AddMember("start_region_id", track_info_ptr->start_region_index, allocator);
    jsonDoc.AddMember("end_region_id", track_info_ptr->leave_region_index, allocator);
    jsonDoc.AddMember("start_frame_id", track_info_ptr->start_frame_index, allocator);
    jsonDoc.AddMember("end_frame_id", track_info_ptr->end_frame_index, allocator);

    rapidjson::Value is_box_array(rapidjson::kArrayType);//创建一个Array类型的元素
    rapidjson::Value is_feature_array(rapidjson::kArrayType);//创建一个Array类型的元素
    rapidjson::Value box_array(rapidjson::kArrayType);//创建一个Array类型的元素
    rapidjson::Value feature_array(rapidjson::kArrayType);//创建一个Array类型的元素
    rapidjson::Value go_through_array(rapidjson::kArrayType);//创建一个Array类型的元素

    std::map<int,bool>::iterator iter;
    for(iter = track_info_ptr->is_box.begin();iter!= track_info_ptr->is_box.end();iter++)
    {
        if(iter->second)
        {
            cv::Rect box = track_info_ptr->track_box_list[iter->first];
            is_box_array.PushBack(have_box, allocator);
            rapidjson::Value boxArray(rapidjson::kArrayType);//创建一个Array类型的元素
            boxArray.PushBack(box.x, allocator);
            boxArray.PushBack(box.y, allocator);
            boxArray.PushBack(box.width, allocator);
            boxArray.PushBack(box.height, allocator);
            box_array.PushBack(boxArray,allocator);

        }
        else
        {
            is_box_array.PushBack(not_box, allocator);
        }
    }
    jsonDoc.AddMember("is_box", is_box_array, allocator);
    jsonDoc.AddMember("box_list", box_array, allocator);

    for(iter = track_info_ptr->is_feature.begin();iter != track_info_ptr->is_feature.end();iter++)
    {
        if(iter->second)
        {
            is_feature_array.PushBack(have_feature,allocator);
            rapidjson::Value boxfeature(rapidjson::kArrayType);//创建一个Array类型的元素

            std::map<int,FEATURE>::iterator res =track_info_ptr->box_feature.find(iter->first);
            if(res == track_info_ptr->box_feature.end())
            {
                std::cout<<"track id: "<<track_info_ptr->track_id<<std::endl;
                std::cout<<"boxfeature size: "<<boxfeature.Size()<<std::endl;
                std::cout<<"fail to find current frame feature"<<std::endl;
                exit(0); 
            }
            
            
            if(track_info_ptr->box_feature[iter->first].rows() == 0)
            {
                std::cout<<"track id: "<<track_info_ptr->track_id<<std::endl;
                std::cout<<"error******************************ERROR"<<std::endl;
                std::cout<<"boxfeature size: "<<boxfeature.Size()<<std::endl;
                exit(0);
            }
            if(track_info_ptr->box_feature[iter->first].cols() != 2048)
            {
                std::cout<<"track id: "<<track_info_ptr->track_id<<std::endl;
                std::cout<<"boxfeature size: "<<boxfeature.Size()<<std::endl;
                std::cout<<"error************track_info_ptr->box_feature[iter->first].rows()******************ERROR"<<std::endl;
                exit(0);
            }

            for(int i = 0; i < track_info_ptr->box_feature[iter->first].rows(); i++) 
            {
                for(int j = 0; j < track_info_ptr->box_feature[iter->first].cols(); j++)
                {
                    boxfeature.PushBack(track_info_ptr->box_feature[iter->first](i,j), allocator);
                } 
            }
            feature_array.PushBack(boxfeature,allocator);
        }
        else
        {
            is_feature_array.PushBack(not_feature,allocator);
        }
    }
    jsonDoc.AddMember("is_feature", is_feature_array, allocator);
    jsonDoc.AddMember("feature_list", feature_array, allocator);

    int last_index = -2;
    for(size_t i =0; i< track_info_ptr->go_through_region_index.size();i++)
    {
        int cur_region_idx = track_info_ptr->go_through_region_index[i];
        if(cur_region_idx != last_index)
        {
            go_through_array.PushBack(cur_region_idx,allocator);
            last_index = cur_region_idx;
        }
    }
    jsonDoc.AddMember("go_through_region", go_through_array, allocator);

    std::string file_name = save_path+ m_city_track_ptr->local_param.cam_name +"/traj_result/"+std::to_string(track_info_ptr->track_id) + ".json";
    std::ofstream outfile;
	outfile.open(file_name.c_str());
	if (!outfile.is_open()) {
		fprintf(stderr, "fail to open file to write: %s\n", file_name.c_str());
		return false;
	}
    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    jsonDoc.Accept(writer);
    std::string strJson = std::string( buffer.GetString());
	outfile << strJson << std::endl;
	outfile.close();
    return true;
}

void tracker::WriteRun()
{
    while(is_running_)
    {
        std::shared_ptr<TRACK_INFO> track_info_p = nullptr;
        track_info_p.reset();
        {
            std::lock_guard<std::mutex> lock(track_info_info_mtx_);
            if(track_info_buff.size() > 0)
            {
                track_info_p = track_info_buff.front();
                track_info_buff.erase(track_info_buff.begin());
            }
            // else
            // {
            //     std::this_thread::sleep_for(std::chrono::milliseconds(100));
            //     continue;
            // } 
        }
        if(track_info_p)
        {
            bool ret = WriteToLocal(track_info_p);
            if(ret)
            {
                std::lock_guard<std::mutex> lock(track_info_obj_mtx);
                for(size_t i = 0; i < remove_tracks.size();i++)
                {
                    if(remove_tracks[i].track_id == track_info_p->track_id)
                    {
                        remove_tracks.erase(remove_tracks.begin()+i);
                        break;
                    }
                }
                std::cout<<"succeed to save file of track_id: "<<track_info_p->track_id<<std::endl;
            }
            else
            {
                std::lock_guard<std::mutex> lock(track_info_obj_mtx);
                for(size_t i = 0; i < remove_tracks.size();i++)
                {
                    if(remove_tracks[i].track_id == track_info_p->track_id)
                    {
                        remove_tracks.erase(remove_tracks.begin()+i);
                        break;
                    }
                }
                
                std::cerr<<"fail to save file of track_id: "<<track_info_p->track_id<<std::endl;
            }
        }
        else
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        
    }
}


double tracker::cal_iou(cv::Rect& box1,cv::Rect& box2)
{
    cv::Rect rect_union = box1 | box2;

    if(rect_union.area() == 0)
    {
        std::cerr<<"WARNING:rect_union.area()==0"<<std::endl;
        return 0.0;
    }

    cv::Rect intersetion = box1 & box2;

     if(intersetion.area() == 0)
    {
        return 0.0;
    }

    double IOU = intersetion.area() *1.0/ rect_union.area();

    return IOU;
}

void tracker::predict(cv::Mat &frame)
{    
    for(Track& track:tracks) 
    {
        track.predit(kf,frame);
    }
}

bool tracker::is_avaiable_save(cv::Rect &save_box,vector<MATCH_DATA>& matches,float threshold)
{
    for(MATCH_DATA& data:matches) 
    {
        int track_idx = data.first;
        double iou_score = cal_iou(save_box,this->tracks[track_idx].track_box);
        if(iou_score > threshold)
        {
            return false;
        }
    }
    
    return true;
}

bool tracker::total_interaction(const DETECTION_ROW &detection)
{
    int total_interaction_eare = 1;
    DETECTBOX tmpbox_ = detection.tlwh;
    cv::Rect box(tmpbox_(0), tmpbox_(1), tmpbox_(2), tmpbox_(3));
    if(box.area() < 400)
    {
        return false;
    }
    for(Track& track:tracks) 
    {
        if(track.time_since_update == 0)
        {
            cv::Rect intersetion = box & track.track_box;
            float inter_score_temp = cal_iou(box,track.track_box);
            if(inter_score_temp > 0.15)
            {
                float self_inter = intersetion.area()*1.0/track.track_box.area()*1.0;
                if(self_inter > 0.47)
                {
                    return false;
                }
                float self_inter_ = intersetion.area()*1.0/box.area()*1.0;
                if(self_inter_ > 0.47)
                {
                    return false;
                }
            }
            total_interaction_eare += intersetion.area();
        }
    }
    float inter_score = total_interaction_eare*1.0/box.area()*1.0;
    if(inter_score > 0.55)
    {
        return false;
    }
    return true;
    
}

void tracker::update_all_track()
{
    for(size_t i = 0; i < tracks.size();i++)
    {
        update_traj_info(i);
    }
}

void tracker::save_track(int track_idx,vector<MATCH_DATA> &matches)
{
    if(tracks[track_idx].eco_predict_status)
    {
       
        if(is_avaiable_save(tracks[track_idx].eco_rect,matches,0.5))
        {
            cv::Rect eco_rect = tracks[track_idx].eco_rect;
            DETECTBOX ret = DETECTBOX(eco_rect.x, eco_rect.y,eco_rect.width,  eco_rect.height);
            DETECTION_ROW temp;
            temp.tlwh = ret;
            tracks[track_idx].pre_rect = eco_rect;
            tracks[track_idx].status = SAVE_ECO;
            tracks[track_idx].track_box = eco_rect;
            tracks[track_idx].update(this->kf, temp);
            this->tracks[track_idx].save_num++;
            
        }
        else
        {
            
            if(tracks[track_idx].optical_predict_status)
            {
                
                if(is_avaiable_save(tracks[track_idx].optical_rect,matches,0.5))
                {
                    cv::Rect optical_rect = tracks[track_idx].optical_rect;
                    tracks[track_idx].pre_rect = optical_rect;
                    tracks[track_idx].status = SAVE_OPTICAL;
                    tracks[track_idx].track_box = optical_rect;
                    DETECTBOX ret = DETECTBOX(optical_rect.x, optical_rect.y,optical_rect.width,  optical_rect.height);
                    DETECTION_ROW temp;
                    temp.tlwh = ret;
                    tracks[track_idx].update(this->kf, temp);
                    this->tracks[track_idx].save_num++;
                }
                else
                {
                    
                    tracks[track_idx].status = MISS;
                    this->tracks[track_idx].mark_missed();
                }
            }
            else
            {
                
                tracks[track_idx].status = MISS;
                this->tracks[track_idx].mark_missed();
            }
        }
    }
    else if(tracks[track_idx].optical_predict_status)
    {
        if(is_avaiable_save(tracks[track_idx].optical_rect,matches,0.4))
        {
            
            cv::Rect optical_rect = tracks[track_idx].optical_rect;
            tracks[track_idx].pre_rect = optical_rect;
            tracks[track_idx].status = SAVE_OPTICAL;
            tracks[track_idx].track_box = optical_rect;
            DETECTBOX ret = DETECTBOX(optical_rect.x, optical_rect.y,optical_rect.width,  optical_rect.height);
            DETECTION_ROW temp;
            temp.tlwh = ret;
            tracks[track_idx].update(this->kf, temp);
            this->tracks[track_idx].save_num++;
        }
        else
        {
            
            tracks[track_idx].status = MISS;
            this->tracks[track_idx].mark_missed();
        }
    }
    else
    {
       
        tracks[track_idx].status = MISS;
        this->tracks[track_idx].mark_missed();
    }
}

void tracker::update(const DETECTIONS &match_detections,const DETECTIONS &create_detections,
cv::Mat&frame,int frame_id)
{
    //DETECTIONS low_threshold_detect;
    current_frame_id = frame_id;
    if(frame.empty())
    {
        std::cerr<<"image is empty fail to update"<<std::endl;
        return;
    }
    //std::cout<<"track size: "<<tracks.size()<<std::endl;
    if(tracks.size() == 0)
    {
        for(size_t i = 0; i < create_detections.size();i++)
        {
            DETECTION_ROW temp = create_detections[i];
            DETECTBOX tmpbox = temp.tlwh;
            cv::Rect detect_bbox(tmpbox(0), tmpbox(1), tmpbox(2), tmpbox(3));
            if(m_city_track_ptr->is_available_create(detect_bbox))
            {
                this->_initiate_track(create_detections[i],frame);
            }
        }
        update_all_track();
        vector<int> active_targets;
        vector<TRACKER_DATA> tid_features;
        for (Track& track:tracks)       
        {
            if(track.time_since_update >= 1)
            {
                continue;
            }
            if(track.status == LOW_DET_M)
            {
                continue;
            }
            if(track.status == SAVE_ECO)
            {
                continue;
            }
            if(track.status == SAVE_OPTICAL)
            {
                continue;
            }
            if(track.status == MISS)
            {
                continue;
            }
            active_targets.push_back(track.track_id);
            tid_features.push_back(std::make_pair(track.track_id, track.features));
            FEATURESS t = FEATURESS(0, 2048);
            track.features = t;
        }
        this->metric->partial_fit(tid_features, active_targets);
        return;
    }

    cv::Mat image = frame.clone();

    TRACHER_MATCHD res_create_detections;  //create_detections

    TRACHER_MATCHD res_match_detections;  //create_detections

    _match(match_detections,create_detections,res_create_detections,res_match_detections,image);

    vector<MATCH_DATA>& matches_c = res_create_detections.matches;
    
    std::vector<int> bb;
    for(MATCH_DATA& data:matches_c) 
    {
        int track_idx = data.first;
        std::vector<int>::iterator iter = std::find(bb.begin(),bb.end(),track_idx);
        if(iter != bb.end())
        {
            std::cout<<"find double "<<track_idx<<std::endl;
            exit(0);
        }
        bb.push_back(track_idx);
        int detection_idx = data.second;
        tracks[track_idx].update(this->kf, create_detections[detection_idx],true);
        this->tracks[track_idx].save_num = 0;
    }

    vector<MATCH_DATA>& matches_m = res_match_detections.matches;
    for(MATCH_DATA& data:matches_m) 
    {
        int track_idx = data.first;
        std::vector<int>::iterator iter = std::find(bb.begin(),bb.end(),track_idx);
        if(iter != bb.end())
        {
            std::cout<<"****find double "<<track_idx<<std::endl;
            exit(0);
        }
        
        int detection_idx = data.second;
        tracks[track_idx].update(this->kf, match_detections[detection_idx],true);
        this->tracks[track_idx].save_num = 0;
    }

    vector<MATCH_DATA> matches;
    matches.assign(matches_c.begin(),matches_c.end());
    matches.insert(matches.end(),matches_m.begin(),matches_m.end());

    vector<int>& unmatched_tracks = res_match_detections.unmatched_tracks;
    for(int& track_idx:unmatched_tracks) 
    {
        if(this->tracks[track_idx].time_since_update > 1)  
        {
            tracks[track_idx].status = MISS;
            this->tracks[track_idx].mark_missed();
        }
        else
        {
            if(tracks[track_idx].track_infomation->traj_status == READY_LEAVE)
            {
                this->tracks[track_idx].status = MISS;
                this->tracks[track_idx].mark_missed();
            }
            else
            {
                if(tracks[track_idx].track_infomation->traj_status == LEAVE)
                {
                   this->tracks[track_idx].set_deleted();
                }
                else
                {
                    if(this->tracks[track_idx].save_num > 5)
                    {
                        this->tracks[track_idx].status = MISS;
                        this->tracks[track_idx].mark_missed();
                    }
                    else
                    {
                        save_track(track_idx,matches);
                    }
                }
            }
            
        }   
    }
    vector<int>& unmatched_detections = res_create_detections.unmatched_detections;
    for(int& detection_idx:unmatched_detections) 
    {
        DETECTION_ROW temp = create_detections[detection_idx];
        DETECTBOX tmpbox = temp.tlwh;
        cv::Rect detect_bbox(tmpbox(0), tmpbox(1), tmpbox(2), tmpbox(3));
        if(m_city_track_ptr->is_available_create(detect_bbox))
        {
            if(total_interaction(create_detections[detection_idx]))
            {
                this->_initiate_track(create_detections[detection_idx],frame);
            }  
        }
    }
    update_all_track();
    if(m_city_track_ptr->local_param.is_save_image)
    {
        draw_result(match_detections,create_detections,frame);
    }
    
    vector<Track>::iterator it;
    for(it = tracks.begin(); it != tracks.end();) 
    {
        
        if((*it).is_deleted())
        {
            //std::cout<<"remove track id:"<<(*it).track_id<<std::endl;
            {
                std::lock_guard<std::mutex> lock(track_info_obj_mtx);
                remove_tracks.push_back((*it));
            }
            it = tracks.erase(it);
        } 
        else
        {
            ++it;
        }
        
    }
    
    vector<int> active_targets;
    vector<TRACKER_DATA> tid_features;
    for (Track& track:tracks)       
    {
        if(track.time_since_update >= 1)
        {
            continue;
        }
        if(track.status == LOW_DET_M)
        {
            continue;
        }
        if(track.status == SAVE_ECO)
        {
            continue;
        }
        if(track.status == SAVE_OPTICAL)
        {
            continue;
        }
        if(track.status == MISS)
        {
            continue;
        }
        active_targets.push_back(track.track_id);
        tid_features.push_back(std::make_pair(track.track_id, track.features));
        FEATURESS t = FEATURESS(0, 2048);
        track.features = t;
    }
    this->metric->partial_fit(tid_features, active_targets);
}

bool tracker::is_avaliable_match(int track_id,cv::Rect&track_box,cv::Rect&detect_box,bool is_cascade)
{
    if(m_city_track_ptr->local_param.cam_name == "c041")
    {
        cv::Rect rect = detect_box;
        bool ret_t = m_city_track_ptr->is_crowd_region(track_id);
        bool ret_d = m_city_track_ptr->is_crowd_region(rect);
        if(ret_t&&ret_d)
        {
            if(is_cascade)
            {
                if((track_box.x- detect_box.x) >= -24)
                {
                    return true;
                }
                else
                {
                    return false;
                }
            }
            else
            {
                if((track_box.x- detect_box.x) >= -25)
                {
                    if(abs(detect_box.y - track_box.y) <= 9)
                    {
                        return true;
                    }
                    else
                    {
                        return false;
                    }
                    
                }
                else
                {
                    return false;
                }
            }
        }
        // else if((ret_t == false) && (ret_d == true))
        // {
        //     return false;
        // }
        // else if((ret_t == false) && (ret_d == false))
        // {
        //     return true;
        // }
        else
        {
            return true;
        }
    }

    // if(m_city_track_ptr->local_param.cam_name == "c042")
    // {
    //     bool ret = m_city_track_ptr->is_crowd_region2(track_id);
    //     if(ret)
    //     {
    //         cv::Rect rect = detect_box;
    //         bool ret_ = m_city_track_ptr->is_contain_crowd_region(rect);
    //         if(!ret_)
    //         {
    //             return false;
    //         }
    //     }
    // }
    return true;
    
}
void tracker::_match(const DETECTIONS &match_detections,const DETECTIONS &create_detections, 
TRACHER_MATCHD &res_create_detections, TRACHER_MATCHD &res_match_detections,cv::Mat &frame)
{
    
    vector<int> confirmed_tracks;
    vector<int> unconfirmed_tracks;
    int idx = 0;
    for(Track& t:tracks) 
    {
        if(t.is_confirmed())
        {
            confirmed_tracks.push_back(idx);
        } 
        else
        {
            unconfirmed_tracks.push_back(idx);
        } 
        idx++;
    }
    // std::cout<<"match_detections: "<<match_detections.size()<<std::endl;
    // std::cout<<"create_detections: "<<create_detections.size()<<std::endl;
    // std::cout<<"confirmed_tracks: "<<confirmed_tracks.size()<<std::endl;
    // std::cout<<"unconfirmed_tracks: "<<unconfirmed_tracks.size()<<std::endl;
    //CASCADE MATCH
    std::vector<int> detection_indices;
    TRACHER_MATCHD matcha = linear_assignment_ptr->matching_cascade(
                this, &tracker::gated_matric,&tracker::optical_iou_cost,&tracker::eco_iou_cost,
                this->metric->mating_threshold,
                1,
                this->tracks,
                create_detections,
                confirmed_tracks,detection_indices,frame,m_merger_distance);
    
    // std::cout<<"matching_cascade size:"<<matcha.matches.size()<<std::endl;

    // std::cout<<"matching_cascade unmatched_detections size:"<<matcha.unmatched_detections.size()<<std::endl;

    // std::cout<<"matching_cascade unmatched_tracks size:"<<matcha.unmatched_tracks.size()<<std::endl;

    //IOU MATCH
    vector<int> iou_track_candidates;
    iou_track_candidates.assign(unconfirmed_tracks.begin(), unconfirmed_tracks.end());
    vector<int>::iterator it;		
    for(it = matcha.unmatched_tracks.begin(); it != matcha.unmatched_tracks.end();) 
    {
        int idx = *it;
        if(tracks[idx].time_since_update == 1) //push into unconfirmed
        { 
            iou_track_candidates.push_back(idx);
            it = matcha.unmatched_tracks.erase(it);
            continue;
        }
        ++it;
    }

    // std::cout<<"matching_cascade size:"<<matcha.matches.size()<<std::endl;

    // std::cout<<"matching_cascade unmatched_detections size:"<<matcha.unmatched_detections.size()<<std::endl;

    // std::cout<<"matching_cascade unmatched_tracks size:"<<matcha.unmatched_tracks.size()<<std::endl;

    // std::cout<<"start to iou match"<<std::endl;
    TRACHER_MATCHD matchb = linear_assignment_ptr->min_cost_matching(
                this, &tracker::iou_cost,&tracker::optical_iou_cost,&tracker::eco_iou_cost,
                this->max_iou_distance,
                this->tracks,
                create_detections,
                iou_track_candidates,
                matcha.unmatched_detections);
    
    
    // std::cout<<"matchb_matching size:"<<matchb.matches.size()<<std::endl;

    // std::cout<<"matchb_matching unmatched_detections size:"<<matchb.unmatched_detections.size()<<std::endl;

    // std::cout<<"matchb_matching unmatched_tracks size:"<<matchb.unmatched_tracks.size()<<std::endl;

    //for(MATCH_DATA& pa:matchb.matches)
    std::vector<MATCH_DATA>::iterator iter_m;
    for(iter_m = matchb.matches.begin();iter_m != matchb.matches.end();)
    {
        int track_idx = iter_m->first;
        int detection_idx = iter_m->second;
        cv::Rect track_box = this->tracks[track_idx].track_box;
        DETECTION_ROW detection_t = create_detections[detection_idx];
        DETECTBOX tmpbox = detection_t.tlwh;
        cv::Rect detect_rect(tmpbox(0), tmpbox(1), tmpbox(2), tmpbox(3));
        bool ret = is_avaliable_match(track_idx,track_box,detect_rect,false);
        if(!ret)
        {
            matchb.unmatched_tracks.push_back(track_idx);
            matchb.unmatched_detections.push_back(detection_idx);
            iter_m = matchb.matches.erase(iter_m);
        }
        else
        {
            iter_m++;
        }
    }

    for(MATCH_DATA& data:matchb.matches) 
    {
        int track_idx = data.first;
        int detection_idx = data.second;
        DETECTBOX tmpbox = create_detections[detection_idx].tlwh;
        cv::Rect2f current_rect(tmpbox(0), tmpbox(1), tmpbox(2), tmpbox(3));
        cv::Rect current_box(tmpbox(0), tmpbox(1), tmpbox(2), tmpbox(3));
        tracks[track_idx].pre_rect = current_rect;
        tracks[track_idx].status = IOUMATCH;
        tracks[track_idx].track_box = current_box;
        double iou_score = linear_assignment_ptr->bbox_iou(tracks[track_idx],
        create_detections[detection_idx]);
        if(iou_score < 0.5)
        {
            tracks[track_idx].init_eco_track(frame);
        }
    }

    //get result:
    res_create_detections.matches.assign(matcha.matches.begin(), matcha.matches.end());
    res_create_detections.matches.insert(res_create_detections.matches.end(),
    matchb.matches.begin(), matchb.matches.end());
     
    //unmatched_tracks;
    res_create_detections.unmatched_tracks.assign(
                matcha.unmatched_tracks.begin(),
                matcha.unmatched_tracks.end());

    res_create_detections.unmatched_tracks.insert(
                res_create_detections.unmatched_tracks.end(),
                matchb.unmatched_tracks.begin(),
                matchb.unmatched_tracks.end());

    res_create_detections.unmatched_detections.assign(
                matchb.unmatched_detections.begin(),
                matchb.unmatched_detections.end());

    TRACHER_MATCHD matchb_1 = linear_assignment_ptr->matching_cascade(
                this, &tracker::gated_matric,&tracker::optical_iou_cost,&tracker::eco_iou_cost,
                this->metric->mating_threshold,
                this->max_age,
                this->tracks,
                create_detections,
                res_create_detections.unmatched_tracks,res_create_detections.unmatched_detections,frame,m_merger_distance,true);

    
    
     //unmatched_tracks;
    res_create_detections.matches.clear();
    res_create_detections.matches.assign(matcha.matches.begin(), matcha.matches.end());

    res_create_detections.matches.insert(res_create_detections.matches.end(),
    matchb.matches.begin(), matchb.matches.end());

    res_create_detections.matches.insert(res_create_detections.matches.end(),
    matchb_1.matches.begin(), matchb_1.matches.end());

    res_create_detections.unmatched_tracks.clear();
    res_create_detections.unmatched_tracks.assign(
                matchb_1.unmatched_tracks.begin(),
                matchb_1.unmatched_tracks.end());

    res_create_detections.unmatched_detections.clear();
    res_create_detections.unmatched_detections.assign(
                matchb_1.unmatched_detections.begin(),
                matchb_1.unmatched_detections.end());

    std::vector<int> iou_temp;
    std::vector<int>::iterator iter_m_p;
    for(iter_m_p = res_create_detections.unmatched_tracks.begin();iter_m_p != res_create_detections.unmatched_tracks.end();)
    {
        int idx = *iter_m_p;
        if(this->tracks[idx].time_since_update == 1)
        {
            iou_temp.push_back(idx);
            iter_m_p = res_create_detections.unmatched_tracks.erase(iter_m_p);
            continue;
        }
        iter_m_p++;
    }
   
    //低域直匹配
    //std::cout<<"start to low threshold match"<<std::endl;
    std::vector<int> low_detection_indices;
    for(size_t i = 0; i < match_detections.size();i++)
    {
        low_detection_indices.push_back(int(i));
    }

    TRACHER_MATCHD matchc = linear_assignment_ptr->min_cost_matching(
                this, &tracker::iou_cost,&tracker::optical_iou_cost,&tracker::eco_iou_cost,
                this->max_iou_distance,
                this->tracks,
                match_detections,
                iou_temp,
                low_detection_indices);

    // std::cout<<"matchc size:"<<matchc.matches.size()<<std::endl;

    // std::cout<<"matchc unmatched_detections size:"<<matchc.unmatched_detections.size()<<std::endl;

    // std::cout<<"matchc unmatched_tracks size:"<<matchc.unmatched_tracks.size()<<std::endl;

    std::vector<MATCH_DATA>::iterator iter_m_c;
    for(iter_m_c = matchc.matches.begin();iter_m_c != matchc.matches.end();)
    {
        int track_idx = iter_m_c->first;
        int detection_idx = iter_m_c->second;
        cv::Rect track_box = this->tracks[track_idx].track_box;
        DETECTION_ROW detection_t = match_detections[detection_idx];
        DETECTBOX tmpbox = detection_t.tlwh;
        cv::Rect detect_rect(tmpbox(0), tmpbox(1), tmpbox(2), tmpbox(3));
        bool ret = is_avaliable_match(track_idx,track_box,detect_rect,false);
        if(!ret)
        {
            matchc.unmatched_tracks.push_back(track_idx);
            matchc.unmatched_detections.push_back(detection_idx);
            iter_m_c = matchc.matches.erase(iter_m_c);
        }
        else
        {
            iter_m_c++;
        }
    }

    for(MATCH_DATA& data:matchc.matches) 
    {
        int track_idx = data.first;
        int detection_idx = data.second;
        DETECTBOX tmpbox = match_detections[detection_idx].tlwh;
        cv::Rect2f current_rect(tmpbox(0), tmpbox(1), tmpbox(2), tmpbox(3));
        cv::Rect current_box(tmpbox(0), tmpbox(1), tmpbox(2), tmpbox(3));
        if(match_detections[detection_idx].is_feature)
        {
            tracks[track_idx].pre_rect = current_rect;
            tracks[track_idx].status = LOW_DET_M_FEATURE;
            tracks[track_idx].track_box = current_box;
            double iou_score = linear_assignment_ptr->bbox_iou(tracks[track_idx],match_detections[detection_idx]);
            if(iou_score < 0.5)
            {
                tracks[track_idx].init_eco_track(frame);
            }
        }
        else
        {
            tracks[track_idx].pre_rect = current_rect;
            tracks[track_idx].status = LOW_DET_M;
            tracks[track_idx].track_box = current_box;
        }
    }

    res_match_detections.matches.assign(matchc.matches.begin(), matchc.matches.end());

    res_match_detections.unmatched_detections.assign(
                matchc.unmatched_detections.begin(),
                matchc.unmatched_detections.end());

    res_match_detections.unmatched_tracks.assign(
                res_create_detections.unmatched_tracks.begin(),
                res_create_detections.unmatched_tracks.end());

    res_match_detections.unmatched_tracks.insert(
                res_match_detections.unmatched_tracks.end(),
                matchc.unmatched_tracks.begin(),
                matchc.unmatched_tracks.end());
}

void tracker::get_low_threshold(const DETECTIONS &match_detections,const DETECTIONS &create_detections,
DETECTIONS &low_threshold_detections)
{
    for(size_t i = 0; i < match_detections.size();i++)
    {
        bool is_low_box = true;
        DETECTION_ROW low_dec = match_detections[i];
        DETECTBOX tmpbox = low_dec.tlwh;
        cv::Rect low_detect_bbox(tmpbox(0), tmpbox(1), tmpbox(2), tmpbox(3));
        for(size_t j = 0; j < create_detections.size();j++)
        {
            DETECTION_ROW high_dec = create_detections[i];
            DETECTBOX tmpbox_ = high_dec.tlwh;
            cv::Rect high_detect_bbox(tmpbox_(0), tmpbox_(1), tmpbox_(2), tmpbox_(3));
            if(cal_iou(low_detect_bbox,high_detect_bbox) > 0.95)
            {
                is_low_box = false;
                break;
            }
        }

        if(is_low_box)
        {
            low_threshold_detections.push_back(low_dec);
        }
    }
    
}
void tracker::draw_result(const DETECTIONS &match_detect_b,const DETECTIONS &create_detect_box,cv::Mat&frame)
{
    
    std::map<int,std::string> traj_s;
    std::map<int,cv::Rect> result;
    std::map<int,MATCH_STATUS> result_type;
    int frame_id = current_frame_id;
    cv::Mat test_show = frame.clone();
    for(Track& track : this->tracks) 
    {
        if(track.time_since_update > 1)
        {
            continue;
        } 

        if(track.track_infomation->traj_status == INIT)
        {
            traj_s[track.track_id] = "-i";
        }
        else if(track.track_infomation->traj_status == ENTER)
        {
            traj_s[track.track_id] = "-en";
        }
        else if(track.track_infomation->traj_status == READY_LEAVE)
        {
            traj_s[track.track_id] = "-rl";
        }
        else if(track.track_infomation->traj_status == LEAVE)
        {
            traj_s[track.track_id] = "-l";
        }
        else
        {
            traj_s[track.track_id] = "-error";
        }
        result.insert(std::make_pair(track.track_id, track.track_box));
        result_type.insert(std::make_pair(track.track_id, track.status));
        cv::Rect optical_rect = track.optical_rect;
        rectangle(test_show, optical_rect, cv::Scalar(255, 0, 0), 1.5);
        std::string label = cv::format("%d", track.track_id);
        cv::putText(test_show, label+"-op", cv::Point(optical_rect.x, optical_rect.y), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 0, 0), 1);

        cv::Rect eco_rect = track.eco_rect;
        rectangle(test_show, eco_rect, cv::Scalar(0, 0, 255), 1.5);
        label = cv::format("%d", track.track_id);
        cv::putText(test_show, label+"-ec", cv::Point(eco_rect.x, eco_rect.y), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 1);
    }

    cv::imwrite(save_path+m_city_track_ptr->local_param.cam_name+"/mid_result/"+std::to_string(frame_id)+".jpg"
    ,test_show);

    cv::Mat dtect_show = frame.clone();
    for(unsigned int k = 0; k < create_detect_box.size(); k++)
    {
        DETECTBOX tmpbox = create_detect_box[k].tlwh;
        cv::Rect rect(tmpbox(0), tmpbox(1), tmpbox(2), tmpbox(3));
        cv::rectangle(dtect_show, rect, cv::Scalar(255,0,0), 4);
        
    }
    for(unsigned int k = 0; k < match_detect_b.size(); k++)
    {
        DETECTBOX tmpbox = match_detect_b[k].tlwh;
        cv::Rect rect(tmpbox(0), tmpbox(1), tmpbox(2), tmpbox(3));
        cv::rectangle(dtect_show, rect, cv::Scalar(0,0,255), 2); 
    }

    cv::imwrite(save_path+m_city_track_ptr->local_param.cam_name+"/detect_result/"+std::to_string(frame_id)+".jpg"
    ,dtect_show);

    std::map<int,cv::Rect>::iterator iter;
    for(iter=result.begin(); iter != result.end();iter++)
    {
        int k = iter->first;
        cv::Rect rect = iter->second;
        std::string traj = traj_s[k];
        if(result_type[k] == CASCADE)
        {
            rectangle(frame, iter->second, cv::Scalar(0, 0, 255), 1.5);
            std::string label = cv::format("%d", k);
            cv::putText(frame, label+"-c"+traj, cv::Point(rect.x, rect.y), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 1);
        }
        else if(result_type[k] == IOUMATCH)
        {
            rectangle(frame, iter->second, cv::Scalar(0, 255, 0), 1.5);
            std::string label = cv::format("%d", k);
            cv::putText(frame, label+"-u"+traj, cv::Point(rect.x, rect.y), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 1);
        }
        else if(result_type[k] == SAVE_ECO)
        {
            rectangle(frame, iter->second, cv::Scalar(255, 0, 0), 1.5);
            std::string label = cv::format("%d", k);
            cv::putText(frame, label+"-e"+traj, cv::Point(rect.x, rect.y), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 0, 0), 1);
        }
        else if(result_type[k] == SAVE_OPTICAL)
        {
            rectangle(frame, iter->second, cv::Scalar(124, 124, 0), 1.5);
            std::string label = cv::format("%d", k);
            cv::putText(frame, label+"-p"+traj, cv::Point(rect.x, rect.y), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(124, 124, 0), 1);
        }
        else if(result_type[k] == LOW_DET_M)
        {
            rectangle(frame, iter->second, cv::Scalar(124, 255, 65), 1.5);
            std::string label = cv::format("%d", k);
            cv::putText(frame, label+"-m"+traj, cv::Point(rect.x, rect.y), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(124, 255, 65), 1);
        }
        else if(result_type[k] == LOW_DET_M_FEATURE)
        {
            rectangle(frame, iter->second, cv::Scalar(255, 255, 65), 1.5);
            std::string label = cv::format("%d", k);
            cv::putText(frame, label+"-mm"+traj, cv::Point(rect.x, rect.y), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 124, 0), 1);
        }
        else if(result_type[k] == CREATE_INIT)
        {
            rectangle(frame, iter->second, cv::Scalar(255, 124, 124), 1.5);
            std::string label = cv::format("%d", k);
            cv::putText(frame, label+"-init"+traj, cv::Point(rect.x, rect.y), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 124, 124), 1);
        }
        else
        {
            rectangle(frame, iter->second, cv::Scalar(124, 124, 255), 1.5);
            std::string label = cv::format("%d", k);
            cv::putText(frame, label+"-f", cv::Point(rect.x, rect.y), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(124, 124, 255), 1);
        }
        
    }
    m_city_track_ptr->draw_region(frame);
    cv::imwrite(save_path+m_city_track_ptr->local_param.cam_name+"/track_result/"+std::to_string(frame_id)+".jpg"
    ,frame);
}
void  tracker::update_traj_info(int track_index)
{
    
    int track_id_temp = this->tracks[track_index].track_id;

    if(tracks[track_index].status == CREATE_INIT)
    {
        int regin_index_temp = -1;
        trajectory_status traj_status_temp = m_city_track_ptr->get_traj_status(track_index,
        regin_index_temp);
        
        if(traj_status_temp == INIT)
        {
            this->tracks[track_index].track_infomation->is_box[current_frame_id] = true;
            this->tracks[track_index].track_infomation->is_feature[current_frame_id] = true;
            this->tracks[track_index].track_infomation->track_box_list[current_frame_id] = this->tracks[track_index].track_box;
            int size = this->tracks[track_index].features.rows();
            if(size == 1)
            {
                this->tracks[track_index].track_infomation->box_feature[current_frame_id] = this->tracks[track_index].features.row(size-1);
            }
            else
            {
                std::cout<<m_city_track_ptr->local_param.cam_name+" error: this->tracks[track_index].features.rows(): "<<size<<"  track_id:"<<track_id_temp<<std::endl;
                std::cout<<"leave status=="<<traj_status_temp<<"status: "<<tracks[track_index].status<<std::endl;
                exit(0);
            }

            this->tracks[track_index].track_infomation->track_id = this->tracks[track_index].track_id;
            this->tracks[track_index].track_infomation->start_region_index = regin_index_temp;
            this->tracks[track_index].track_infomation->start_frame_index = current_frame_id;
            this->tracks[track_index].track_infomation->go_through_region_index.push_back(regin_index_temp);


            if(this->current_frame_id == 2000)
            {
                this->tracks[track_index].track_infomation->end_frame_index = current_frame_id;
                this->tracks[track_index].track_infomation->leave_region_index = regin_index_temp;
                this->tracks[track_index].track_infomation->traj_status = LEAVE;
                traj_status_temp = LEAVE;
            }

        }
        else if((traj_status_temp == ENTER)||(traj_status_temp == READY_LEAVE))
        {
            this->tracks[track_index].track_infomation->start_frame_index = current_frame_id;
            this->tracks[track_index].track_infomation->start_region_index = regin_index_temp;
            this->tracks[track_index].track_infomation->go_through_region_index.push_back(regin_index_temp);
            this->tracks[track_index].track_infomation->track_id = this->tracks[track_index].track_id;
            this->tracks[track_index].track_infomation->is_box[current_frame_id] = true;
            this->tracks[track_index].track_infomation->is_feature[current_frame_id] = true;
            this->tracks[track_index].track_infomation->track_box_list[current_frame_id] = this->tracks[track_index].track_box;
            int size = this->tracks[track_index].features.rows();
            if(size == 1)
            {
                this->tracks[track_index].track_infomation->box_feature[current_frame_id] = this->tracks[track_index].features.row(size-1);
            }
            else
            {
                std::cout<<m_city_track_ptr->local_param.cam_name+"error: this->tracks[track_index].features.rows(): "<<size<<"  track_id:"<<track_id_temp<<std::endl;
                std::cout<<"leave status=="<<traj_status_temp<<"status: "<<tracks[track_index].status<<std::endl;
                exit(0);
            }

            if(this->current_frame_id == 2000)
            {
                this->tracks[track_index].track_infomation->end_frame_index = current_frame_id;
                this->tracks[track_index].track_infomation->leave_region_index = regin_index_temp;
                this->tracks[track_index].track_infomation->traj_status = LEAVE;
                traj_status_temp = LEAVE;
                {
                    std::lock_guard<std::mutex> lock(track_info_info_mtx_);
                    track_info_buff.push_back(this->tracks[track_index].track_infomation);
                }
                this->tracks[track_index].set_deleted();
            }
        }
        else
        {
            std::cerr<<"WARNING:CREATE_INIT ERROR this->tracks[track_index]: "<<this->tracks[track_index].track_infomation->traj_status<<std::endl;
        }
    }
    else if(tracks[track_index].status == MISS)
    {
        int regin_index_temp = -1;
        trajectory_status traj_status_temp = m_city_track_ptr->get_traj_status(track_index,regin_index_temp);
        if(this->tracks[track_index].is_confirmed() == false)
        {
            if(regin_index_temp == -1)
            {
                std::vector<int> region_id_list_p = this->tracks[track_index].track_infomation->go_through_region_index;

                
                while(1)
                {
                    if(region_id_list_p.size() == 0)
                    {
                        break;
                    }
                    int tmp_ = region_id_list_p.back();
                    region_id_list_p.pop_back();
                    if(tmp_ != -1)
                    {
                        regin_index_temp = tmp_;
                        break;
                    }

                }

            }
            this->tracks[track_index].track_infomation->leave_region_index = regin_index_temp;
            this->tracks[track_index].track_infomation->end_frame_index = current_frame_id-1;
            this->tracks[track_index].track_infomation->traj_status = LEAVE;
            {
                std::lock_guard<std::mutex> lock(track_info_info_mtx_);
                track_info_buff.push_back(this->tracks[track_index].track_infomation);
            }
            this->tracks[track_index].set_deleted();
            return;
        }
        else
        {
            if((traj_status_temp == READY_LEAVE)||(traj_status_temp == LEAVE))
            {
                if(regin_index_temp == -1)
                {
                    std::vector<int> region_id_list_p = this->tracks[track_index].track_infomation->go_through_region_index;
                    while(1)
                    {
                        if(region_id_list_p.size() == 0)
                        {
                            break;
                        }
                        int tmp_ = region_id_list_p.back();
                        region_id_list_p.pop_back();
                        if(tmp_ != -1)
                        {
                            regin_index_temp = tmp_;
                            break;
                        }

                    }

                }
                this->tracks[track_index].track_infomation->leave_region_index = regin_index_temp;
                this->tracks[track_index].track_infomation->end_frame_index = current_frame_id-1;
                this->tracks[track_index].track_infomation->traj_status = LEAVE;
                {
                    std::lock_guard<std::mutex> lock(track_info_info_mtx_);
                    track_info_buff.push_back(this->tracks[track_index].track_infomation);
                }
                this->tracks[track_index].set_deleted();
                return;
            }
            else
            {
                this->tracks[track_index].track_infomation->is_box[current_frame_id] = false;
                this->tracks[track_index].track_infomation->is_feature[current_frame_id] = false;
            }
        }
        int res = m_city_track_ptr->is_in_leave_region_or_ten(regin_index_temp);
        if(res == 1)
        {
            if(this->tracks[track_index].time_since_update < (this->tracks[track_index]._max_age - 1))
            {
                this->tracks[track_index].time_since_update = (this->tracks[track_index]._max_age - 1);
            }
        }
        if(res == 2)
        {
            if(this->tracks[track_index].time_since_update < (this->tracks[track_index]._max_age - 4))
            {
                this->tracks[track_index].time_since_update = (this->tracks[track_index]._max_age - 4);
            }
        }
        
        if(this->current_frame_id == 2000)
        {
            if(regin_index_temp == -1)
            {
                std::vector<int> region_id_list_p = this->tracks[track_index].track_infomation->go_through_region_index;
                while(1)
                {
                    if(region_id_list_p.size() == 0)
                    {
                        break;
                    }
                    int tmp_ = region_id_list_p.back();
                    region_id_list_p.pop_back();
                    if(tmp_ != -1)
                    {
                        regin_index_temp = tmp_;
                        break;
                    }
                }

            }
            this->tracks[track_index].track_infomation->end_frame_index = current_frame_id;
            this->tracks[track_index].track_infomation->leave_region_index = regin_index_temp;
            this->tracks[track_index].track_infomation->traj_status = LEAVE;
            {
                std::lock_guard<std::mutex> lock(track_info_info_mtx_);
                track_info_buff.push_back(this->tracks[track_index].track_infomation);
            }
            this->tracks[track_index].set_deleted();
        }
    }
    else
    {
        
        
        int regin_index_temp = -1;
        trajectory_status traj_status_temp = m_city_track_ptr->get_traj_status(track_index,regin_index_temp);
        this->tracks[track_index].track_infomation->traj_status = traj_status_temp;
        
        // if(this->tracks[track_index].track_id == 16)
        // {
        //     std::cout<<"current frame: "<<current_frame_id<<"  current region_id:"<<regin_index_temp<<"  current status:"
        //     <<traj_status_temp<<std::endl;
        // }

        if(traj_status_temp == INIT)
        {
            
            if((tracks[track_index].status == CASCADE)||(tracks[track_index].status == IOUMATCH))
            {
                
                this->tracks[track_index].track_infomation->is_box[current_frame_id] = true;
                this->tracks[track_index].track_infomation->is_feature[current_frame_id] = true;
                this->tracks[track_index].track_infomation->track_box_list[current_frame_id] = this->tracks[track_index].track_box;
                int size = this->tracks[track_index].features.rows();
                if(size == 1)
                {
                    this->tracks[track_index].track_infomation->box_feature[current_frame_id] = this->tracks[track_index].features.row(size-1);
                }
                else
                {
                    std::cout<<"LEAVE=="<<traj_status_temp<<"status: "<<tracks[track_index].status<<std::endl;
                    std::cout<<"leave status=="<<traj_status_temp<<"status: "<<tracks[track_index].status<<std::endl;
                    exit(0);
                }

                this->tracks[track_index].track_infomation->track_id = this->tracks[track_index].track_id;
            }
            else
            {
               
                this->tracks[track_index].track_infomation->is_box[current_frame_id] = true;
                this->tracks[track_index].track_infomation->is_feature[current_frame_id] = false;
                this->tracks[track_index].track_infomation->track_box_list[current_frame_id] = this->tracks[track_index].track_box;
            }
            if(this->current_frame_id == 2000)
            {
                if(regin_index_temp == -1)
                {
                    std::vector<int> region_id_list_p = this->tracks[track_index].track_infomation->go_through_region_index;
                    while(1)
                    {
                        if(region_id_list_p.size() == 0)
                        {
                            break;
                        }
                        int tmp_ = region_id_list_p.back();
                        region_id_list_p.pop_back();
                        if(tmp_ != -1)
                        {
                            regin_index_temp = tmp_;
                            break;
                        }

                    }

                }
                this->tracks[track_index].track_infomation->end_frame_index = current_frame_id;
                this->tracks[track_index].track_infomation->leave_region_index = regin_index_temp;
                this->tracks[track_index].track_infomation->traj_status = LEAVE;
                {
                    std::lock_guard<std::mutex> lock(track_info_info_mtx_);
                    track_info_buff.push_back(this->tracks[track_index].track_infomation);
                }
                this->tracks[track_index].set_deleted();
            }
            return;
        }
        else if((traj_status_temp == ENTER)||(traj_status_temp == READY_LEAVE))
        {
            if(this->tracks[track_index].track_infomation->start_region_index == -1)
            {
                this->tracks[track_index].track_infomation->start_region_index = regin_index_temp;
            }

            this->tracks[track_index].track_infomation->go_through_region_index.push_back(regin_index_temp);
            this->tracks[track_index].track_infomation->track_id = this->tracks[track_index].track_id;
            if((tracks[track_index].status == CASCADE)||(tracks[track_index].status == IOUMATCH) || 
            (tracks[track_index].status == LOW_DET_M_FEATURE) )
            {
                this->tracks[track_index].track_infomation->is_box[current_frame_id] = true;
                this->tracks[track_index].track_infomation->is_feature[current_frame_id] = true;
                this->tracks[track_index].track_infomation->track_box_list[current_frame_id] = this->tracks[track_index].track_box;
                int size = this->tracks[track_index].features.rows();
                if(size == 1)
                {
                    this->tracks[track_index].track_infomation->box_feature[current_frame_id] = this->tracks[track_index].features.row(size-1);
                }
                else
                {
                    
                    std::cout<<m_city_track_ptr->local_param.cam_name+" error: this->tracks[track_index].features.rows(): "<<size<<"  track_id:"<<track_id_temp<<std::endl;
                    std::cout<<"leave status=="<<traj_status_temp<<"status: "<<tracks[track_index].status<<std::endl;
                    exit(0);
                }
                
            }
            else
            {
                this->tracks[track_index].track_infomation->is_box[current_frame_id] = true;
                this->tracks[track_index].track_infomation->is_feature[current_frame_id] = false;
                this->tracks[track_index].track_infomation->track_box_list[current_frame_id] = this->tracks[track_index].track_box;
            }
            if(this->current_frame_id == 2000)
            {
                if(regin_index_temp == -1)
                {
                    std::vector<int> region_id_list_p = this->tracks[track_index].track_infomation->go_through_region_index;
                    while(1)
                    {
                        if(region_id_list_p.size() == 0)
                        {
                            break;
                        }
                        int tmp_ = region_id_list_p.back();
                        region_id_list_p.pop_back();
                        if(tmp_ != -1)
                        {
                            regin_index_temp = tmp_;
                            break;
                        }

                    }

                }
                this->tracks[track_index].track_infomation->end_frame_index = current_frame_id;
                this->tracks[track_index].track_infomation->leave_region_index = regin_index_temp;
                this->tracks[track_index].track_infomation->traj_status = LEAVE;
                {
                    std::lock_guard<std::mutex> lock(track_info_info_mtx_);
                    track_info_buff.push_back(this->tracks[track_index].track_infomation);
                }
                this->tracks[track_index].set_deleted();
            }
            return;
        }
        else if(traj_status_temp == LEAVE)
        {
            if(regin_index_temp == -1)
            {
                std::vector<int> region_id_list_p = this->tracks[track_index].track_infomation->go_through_region_index;
                while(1)
                {
                    if(region_id_list_p.size() == 0)
                    {
                        break;
                    }
                    int tmp_ = region_id_list_p.back();
                    region_id_list_p.pop_back();
                    if(tmp_ != -1)
                    {
                        regin_index_temp = tmp_;
                        break;
                    }

                }

            }
            this->tracks[track_index].track_infomation->leave_region_index = regin_index_temp;
            this->tracks[track_index].track_infomation->go_through_region_index.push_back(regin_index_temp);
            this->tracks[track_index].track_infomation->track_id = this->tracks[track_index].track_id;
            this->tracks[track_index].track_infomation->end_frame_index = current_frame_id;
            if((tracks[track_index].status == CASCADE)||(tracks[track_index].status == IOUMATCH)|| 
            (tracks[track_index].status == LOW_DET_M_FEATURE))
            {
                this->tracks[track_index].track_infomation->is_box[current_frame_id] = true;
                this->tracks[track_index].track_infomation->is_feature[current_frame_id] = true;
                this->tracks[track_index].track_infomation->track_box_list[current_frame_id] = this->tracks[track_index].track_box;
                int size = this->tracks[track_index].features.rows();
                if(size == 1)
                {
                    this->tracks[track_index].track_infomation->box_feature[current_frame_id] = this->tracks[track_index].features.row(size-1);
                }
                else
                {
                    
                    std::cout<<m_city_track_ptr->local_param.cam_name+ "error: this->tracks[track_index].features.rows(): "<<size<<"  track_id:"<<track_id_temp<<std::endl;
                    std::cout<<"leave status=="<<traj_status_temp<<"status: "<<tracks[track_index].status<<std::endl;
                    exit(0);
                }
                {
                    std::lock_guard<std::mutex> lock(track_info_info_mtx_);
                    track_info_buff.push_back(this->tracks[track_index].track_infomation);
                }
            }
            else 
            {
                this->tracks[track_index].track_infomation->is_box[current_frame_id] = true;
                this->tracks[track_index].track_infomation->is_feature[current_frame_id] = false;
                this->tracks[track_index].track_infomation->track_box_list[current_frame_id] = this->tracks[track_index].track_box;
                {
                    std::lock_guard<std::mutex> lock(track_info_info_mtx_);
                    track_info_buff.push_back(this->tracks[track_index].track_infomation);
                }
            }
            this->tracks[track_index].set_deleted();
        }
        else
        {
            std::cout<<"WARNING:MATCH OR SAVE ERROR this->tracks[track_index]: "<<this->tracks[track_index].track_infomation->traj_status<<std::endl;
        }
    }
}

void tracker::_initiate_track(const DETECTION_ROW &detection,cv::Mat &frame)
{
    KAL_DATA data = kf->initiate(detection.to_xyah());

    KAL_MEAN mean = data.first;

    KAL_COVA covariance = data.second;

    Track track_obj = Track(mean, covariance, this->_next_idx, this->n_init,this->max_age, detection.feature);

    DETECTBOX tmpbox = detection.tlwh;

    cv::Rect cur_rect(tmpbox(0), tmpbox(1), tmpbox(2), tmpbox(3));

    track_obj.pre_rect = cur_rect;

    track_obj.track_box = cur_rect;

    bool ret = track_obj.init_eco_track(frame);
    if(!ret)
    {
        std::cerr<<"WARNING: Fail to init_eco_track"<<std::endl;
    }
    else
    {
        track_obj.eco_rect = cur_rect;
    }
    
    ret = track_obj.init_optical_track(frame);
    if(!ret)
    {
        std::cerr<<"WARNING: Fail to init_optical_track"<<std::endl;
    }
    else
    {
        track_obj.optical_rect = cur_rect;
    }

    this->tracks.push_back(track_obj);

    _next_idx += 1;
}

DYNAMICM tracker::gated_matric(
        std::vector<Track> &tracks,
        const DETECTIONS &dets,
        const std::vector<int>& track_indices,
        const std::vector<int>& detection_indices)
{
    FEATURESS features(detection_indices.size(), 2048);
    int pos = 0;
    for(int i:detection_indices) {
        features.row(pos++) = dets[i].feature;
    }
    vector<int> targets;
    for(int i:track_indices) {
        targets.push_back(tracks[i].track_id);
    }
    DYNAMICM cost_matrix = this->metric->distance(features, targets);
    DYNAMICM res = linear_assignment_ptr->gate_cost_matrix(
                this->kf, cost_matrix, tracks, dets, track_indices,
                detection_indices);
    return res;
}

DYNAMICM tracker::optical_iou_cost(
        std::vector<Track> &tracks,
        const DETECTIONS &dets,
        const std::vector<int>& track_indices,
        const std::vector<int>& detection_indices)
{
    int rows = track_indices.size();
    int cols = detection_indices.size();
    DYNAMICM cost_matrix = Eigen::MatrixXf::Zero(rows, cols);
    for(int i = 0; i < rows; i++) 
    {
        int track_idx = track_indices[i];
        // if(tracks[track_idx].time_since_update > 1) 
        // {
        //     cost_matrix.row(i) = Eigen::RowVectorXf::Constant(cols, INFTY_COST);
        //     continue;
        // }
        if(tracks[track_idx].optical_predict_status)
        {
            DETECTBOX bbox = tracks[track_idx].optical_to_tlwh();
            int csize = detection_indices.size();
            DETECTBOXSS candidates(csize, 4);
            for(int k = 0; k < csize; k++) 
            {
                candidates.row(k) = dets[detection_indices[k]].tlwh;
            }
            Eigen::RowVectorXf rowV = (1. - iou(bbox, candidates).array()).matrix().transpose();
            cost_matrix.row(i) = rowV;
        }
        else
        {
            cost_matrix.row(i) = Eigen::RowVectorXf::Constant(cols, 1);;
            continue;
        }
        
    }
    return cost_matrix;
}

DYNAMICM tracker::eco_iou_cost(
        std::vector<Track> &tracks,
        const DETECTIONS &dets,
        const std::vector<int>& track_indices,
        const std::vector<int>& detection_indices)
{
    int rows = track_indices.size();
    int cols = detection_indices.size();
    DYNAMICM cost_matrix = Eigen::MatrixXf::Zero(rows, cols);
    for(int i = 0; i < rows; i++) {
        int track_idx = track_indices[i];
        // if(tracks[track_idx].time_since_update > 1) 
        // {
        //     cost_matrix.row(i) = Eigen::RowVectorXf::Constant(cols, INFTY_COST);
        //     continue;
        // }
        // bool optical_predict_status;

        // bool eco_predict_status;
        if(tracks[track_idx].eco_predict_status)
        {
            DETECTBOX bbox = tracks[track_idx].eco_to_tlwh();
            int csize = detection_indices.size();
            DETECTBOXSS candidates(csize, 4);
            for(int k = 0; k < csize; k++) 
            {   
                candidates.row(k) = dets[detection_indices[k]].tlwh;
            }
            Eigen::RowVectorXf rowV = (1. - iou(bbox, candidates).array()).matrix().transpose();
            cost_matrix.row(i) = rowV;
        }
        else
        {
            cost_matrix.row(i) = Eigen::RowVectorXf::Constant(cols, 1);;
            continue;
        }
        
    }
    return cost_matrix;
}



DYNAMICM tracker::iou_cost(
        std::vector<Track> &tracks,
        const DETECTIONS &dets,
        const std::vector<int>& track_indices,
        const std::vector<int>& detection_indices)
{
    //!!!python diff: track_indices && detection_indices will never be None.
    //    if(track_indices.empty() == true) {
    //        for(size_t i = 0; i < tracks.size(); i++) {
    //            track_indices.push_back(i);
    //        }
    //    }
    //    if(detection_indices.empty() == true) {
    //        for(size_t i = 0; i < dets.size(); i++) {
    //            detection_indices.push_back(i);
    //        }
    //    }
    int rows = track_indices.size();
    int cols = detection_indices.size();
    DYNAMICM cost_matrix = Eigen::MatrixXf::Zero(rows, cols);
    for(int i = 0; i < rows; i++) {
        int track_idx = track_indices[i];
        if(tracks[track_idx].time_since_update > 1) {
            cost_matrix.row(i) = Eigen::RowVectorXf::Constant(cols, 1);
            continue;
        }
        // cost_matrix.row(i) = Eigen::RowVectorXf::Constant(cols, 1);;
        // continue;
        DETECTBOX bbox = tracks[track_idx].to_tlwh();
        int csize = detection_indices.size();
        DETECTBOXSS candidates(csize, 4);
        for(int k = 0; k < csize; k++) candidates.row(k) = dets[detection_indices[k]].tlwh;
        Eigen::RowVectorXf rowV = (1. - iou(bbox, candidates).array()).matrix().transpose();
        cost_matrix.row(i) = rowV;
    }
    return cost_matrix;
}

Eigen::VectorXf
tracker::iou(DETECTBOX& bbox, DETECTBOXSS& candidates)
{
    float bbox_tl_1 = bbox[0];
    float bbox_tl_2 = bbox[1];
    float bbox_br_1 = bbox[0] + bbox[2];
    float bbox_br_2 = bbox[1] + bbox[3];
    float area_bbox = bbox[2] * bbox[3];

    Eigen::Matrix<float, -1, 2> candidates_tl;
    Eigen::Matrix<float, -1, 2> candidates_br;
    candidates_tl = candidates.leftCols(2) ;
    candidates_br = candidates.rightCols(2) + candidates_tl;

    int size = int(candidates.rows());
    //    Eigen::VectorXf area_intersection(size);
    //    Eigen::VectorXf area_candidates(size);
    Eigen::VectorXf res(size);
    for(int i = 0; i < size; i++) {
        float tl_1 = std::max(bbox_tl_1, candidates_tl(i, 0));
        float tl_2 = std::max(bbox_tl_2, candidates_tl(i, 1));
        float br_1 = std::min(bbox_br_1, candidates_br(i, 0));
        float br_2 = std::min(bbox_br_2, candidates_br(i, 1));

        float w = br_1 - tl_1; w = (w < 0? 0: w);
        float h = br_2 - tl_2; h = (h < 0? 0: h);
        float area_intersection = w * h;
        float area_candidates = candidates(i, 2) * candidates(i, 3);
        res[i] = area_intersection/(area_bbox + area_candidates - area_intersection);
    }
    //#ifdef MY_inner_DEBUG
    //        std::cout << res << std::endl;
    //#endif
    return res;
}

