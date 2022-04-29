#ifndef TRACK_H
#define TRACK_H

#include "dataType.h"
#include "kalmanfilter.h"
#include "model.h"
#include "MedianFlow.h"
#include "eco.hpp"
#include "define_info.h"
#include<memory>


enum MATCH_STATUS
{
    CREATE_INIT,
    CASCADE,
    IOUMATCH,
    SAVE_ECO,
    SAVE_OPTICAL,
    LOW_DET_M,
    LOW_DET_M_FEATURE,
    MISS
};

enum trajectory_status
{
    INIT,
    ENTER,
    READY_LEAVE,
    LEAVE
};

typedef struct track_
{
    std::map<int,bool> is_box;
    std::map<int,bool>is_feature;
    std::map<int,cv::Rect> track_box_list;
    std::map<int,FEATURE> box_feature;
    std::vector<int> go_through_region_index;

    trajectory_status traj_status;
    int track_id;
    int start_region_index;
    int leave_region_index;
    int start_frame_index;
    int end_frame_index;
    int ready_l = 0;
    track_()
    {
        ready_l = 0;
        traj_status = INIT;
        track_id = -1;
        start_region_index = -1;
        leave_region_index = -1;
        start_frame_index = -1;
        end_frame_index = -1;
    };

}TRACK_INFO;


class Track
{
    /*"""
    A single target track with state space `(x, y, a, h)` and associated
    velocities, where `(x, y)` is the center of the bounding box, `a` is the
    aspect ratio and `h` is the height.

    Parameters
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.
    max_age : int
        The maximum number of consecutive misses before the track state is
        set to `Deleted`.
    feature : Optional[ndarray]
        Feature vector of the detection this track originates from. If not None,
        this feature is added to the `features` cache.

    Attributes
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    hits : int
        Total number of measurement updates.
    age : int
        Total number of frames since first occurance.
    time_since_update : int
        Total number of frames since last measurement update.
    state : TrackState
        The current track state.
    features : List[ndarray]
        A cache of features. On each measurement update, the associated feature
        vector is added to this list.

    """*/
    enum TrackState {Tentative = 1, Confirmed, Deleted};

public:
    Track(KAL_MEAN& mean, KAL_COVA& covariance, int track_id,
          int n_init, int max_age, const FEATURE& feature);

    void predit(KalmanFilterTrack *kf,cv::Mat &frame);

    void update(KalmanFilterTrack * const kf, const DETECTION_ROW& detection,bool is_cascade_match = false);

    void mark_missed();
    bool is_confirmed();
    bool is_deleted();
    bool is_tentative();

    DETECTBOX to_tlwh();

    DETECTBOX optical_to_tlwh();

    DETECTBOX eco_to_tlwh();


    bool set_deleted();

    bool init_optical_track(cv::Mat&frame);

    bool init_eco_track(cv::Mat&frame);

    bool optical_track_predict(cv::Mat&frame);

    bool eco_track_predict(cv::Mat&frame);

    int time_since_update;

    int track_id;

    FEATURESS features;

    KAL_MEAN mean;
    
    KAL_COVA covariance;

    MATCH_STATUS status;

    int hits;
    int age;
    int _n_init;
    int _max_age;

    std::shared_ptr<TRACK_INFO> track_infomation;
 

    cv::Rect track_box;

    TrackState state;

    cv::Rect2f pre_rect;

    bool is_init_optical;

    bool is_init_eco_track;

    cv::Rect optical_rect;

    cv::Rect eco_rect;

    float eco_score;

    float optical_score;

    bool optical_predict_status;

    bool eco_predict_status;

    std::shared_ptr<MedianFlow> optical_track;

    std::shared_ptr<eco::ECO> eco_track;

    cv::Rect predict_box;

    bool is_exist_feature;

    int save_num;

private:
    void featuresAppendOne(const FEATURE& f);
};

#endif // TRACK_H
