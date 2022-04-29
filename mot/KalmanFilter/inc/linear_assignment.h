#ifndef LINEAR_ASSIGNMENT_H
#define LINEAR_ASSIGNMENT_H
#include "dataType.h"
#include "tracker.h"

#define INFTY_COST 1e6
class tracker;
//for matching;
class linear_assignment
{
public:
    linear_assignment(tracker* ptr_t_p);
    linear_assignment(const linear_assignment& );
    linear_assignment& operator=(const linear_assignment&);
    static linear_assignment* instance;

    tracker* ptr_t;


    //static linear_assignment* getInstance();
    TRACHER_MATCHD matching_cascade(tracker* distance_metric,
            tracker::GATED_METRIC_FUNC distance_metric_func_1,
            tracker::GATED_METRIC_FUNC distance_metric_func_2,
            tracker::GATED_METRIC_FUNC distance_metric_func_3,
            float max_distance,
            int cascade_depth,
            std::vector<Track>& tracks,
            const DETECTIONS& detections,
            std::vector<int> &track_indices,
            std::vector<int> &detection_indices,cv::Mat&frame,float merger_distance=0.7,bool is_cascade = false);
    TRACHER_MATCHD min_cost_matching(
            tracker* distance_metric,
            tracker::GATED_METRIC_FUNC distance_metric_func_1,
            tracker::GATED_METRIC_FUNC distance_metric_func_2,
            tracker::GATED_METRIC_FUNC distance_metric_func_3,
            float max_distance,
            std::vector<Track>& tracks,
            const DETECTIONS& detections,
            std::vector<int>& track_indices,
            std::vector<int>& detection_indices,bool is_iou=true,bool is_match_miss= false);
    DYNAMICM gate_cost_matrix(
            KalmanFilterTrack* kf,
            DYNAMICM& cost_matrix,
            std::vector<Track>& tracks,
            const DETECTIONS& detections,
            const std::vector<int>& track_indices,
            const std::vector<int>& detection_indices,
            float gated_cost = INFTY_COST,
            bool only_position = false);
     double bbox_iou(Track& track,const DETECTION_ROW& det);
};

#endif // LINEAR_ASSIGNMENT_H
