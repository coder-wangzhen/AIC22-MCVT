#ifndef TRACKER_H
#define TRACKER_H
#include <vector>

#include "kalmanfilter.h"
#include "track.h"
#include "model.h"
#include <thread>
#include <vector>
#include <memory>
#include "rapidjson/document.h"
#include "rapidjson/writer.h"
#include "rapidjson/stringbuffer.h"
#include "city_tracker.h"



class CityTracker;

class NearNeighborDisMetric;

class linear_assignment;

class tracker
{
public:
    NearNeighborDisMetric* metric;
    float max_iou_distance;
    float m_merger_distance;
    KalmanFilterTrack* kf;
    int max_age;
    int n_init;
    int _next_idx;
    int current_frame_id;
    CityTracker * m_city_track_ptr;
    std::shared_ptr<linear_assignment> linear_assignment_ptr;

    std::thread run_thread_;  // 运行线程
    bool is_running_ = false; // 运行开关
    std::mutex track_info_info_mtx_;
    std::mutex track_info_obj_mtx;
    std::vector<std::shared_ptr<TRACK_INFO>> track_info_buff;

public:
    std::vector<Track> tracks;
    std::vector<Track> remove_tracks;
    tracker(/*NearNeighborDisMetric* metric,*/CityTracker * city_track_ptr,
    		float max_cosine_distance,float merger_distance, int nn_budget,
            float max_iou_distance = 0.7,
            int max_age = 500, int n_init=3);
    ~tracker();
    void predict(cv::Mat &frame);

    void update(const DETECTIONS& detections,const DETECTIONS &create_detections,cv::Mat &frame,int frame_id);

    //bool is_high_threshold_box(const DETECTION_ROW &detection);

    double cal_iou(cv::Rect& box1,cv::Rect& box2);

    bool total_interaction(const DETECTION_ROW &detection);

    bool is_avaiable_save(cv::Rect &save_box,vector<MATCH_DATA>& matches,float threshold);

    void get_low_threshold(const DETECTIONS &match_detections,const DETECTIONS &create_detections,
DETECTIONS &low_threshold_detections);

    void update_all_track();

    void update_traj_info(int track_index);

    bool WriteToLocal(std::shared_ptr<TRACK_INFO> track_info_ptr);

    void WriteRun();

    void save_track(int track_idx,vector<MATCH_DATA> &matches);

    void draw_result(const DETECTIONS &match_detect_b,const DETECTIONS &create_detect_box,cv::Mat&frame);

    typedef DYNAMICM (tracker::* GATED_METRIC_FUNC)(
            std::vector<Track>& tracks,
            const DETECTIONS& dets,
            const std::vector<int>& track_indices,
            const std::vector<int>& detection_indices);

    std::string save_path; 

    bool is_avaliable_match(int track_id,cv::Rect&track_box,cv::Rect&detect_box,bool is_cascade);

private:    

    void _match(const DETECTIONS &match_detections,const DETECTIONS &create_detections, 
TRACHER_MATCHD &res_create_detections, TRACHER_MATCHD &res_match_detections,cv::Mat &frame);

    void _initiate_track(const DETECTION_ROW &detection,cv::Mat &frame);

public:

    DYNAMICM gated_matric(
            std::vector<Track>& tracks,
            const DETECTIONS& dets,
            const std::vector<int>& track_indices,
            const std::vector<int>& detection_indices);
    DYNAMICM iou_cost(
            std::vector<Track>& tracks,
            const DETECTIONS& dets,
            const std::vector<int>& track_indices,
            const std::vector<int>& detection_indices);
    Eigen::VectorXf iou(DETECTBOX& bbox,
            DETECTBOXSS &candidates);
    DYNAMICM optical_iou_cost(
        std::vector<Track> &tracks,
        const DETECTIONS &dets,
        const std::vector<int>& track_indices,
        const std::vector<int>& detection_indices);
    DYNAMICM eco_iou_cost(
        std::vector<Track> &tracks,
        const DETECTIONS &dets,
        const std::vector<int>& track_indices,
        const std::vector<int>& detection_indices);
};

#endif // TRACKER_H
