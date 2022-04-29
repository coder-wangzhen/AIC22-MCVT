#include "track.h"

Track::Track(KAL_MEAN& mean, KAL_COVA& covariance, int track_id, int n_init, int max_age, const FEATURE& feature)
{
    this->mean = mean;

    this->covariance = covariance;

    this->track_id = track_id;

    this->hits = 1;

    this->age = 1;

    this->time_since_update = 0;

    this->state = TrackState::Tentative;

    features = FEATURESS(1, 2048);

    features.row(0) = feature;//features.rows() must = 0;

    this->_n_init = n_init;

    this->_max_age = max_age;

    is_init_optical =false;

    is_init_eco_track= false;

    optical_track = nullptr;

    eco_track = nullptr;

    optical_predict_status = false;

    eco_predict_status = false;

    save_num = 0;

    status = CREATE_INIT;

    track_infomation = std::make_shared<TRACK_INFO>();
}

void Track::predit(KalmanFilterTrack *kf,cv::Mat &frame)
{
    /*Propagate the state distribution to the current time step using a
    Kalman filter prediction step.

    Parameters
    ----------
    kf : kalman_filter.KalmanFilterTrack
        The Kalman filter.
    */
    if(frame.empty())
    {
        std::cout<<"predit: image is empty"<<std::endl;
        exit(0);
    }
    cv::Mat temp = frame.clone();
    bool ret = optical_track_predict(temp);
    if(!ret)
    {
        std::cerr<<"WARNING: fail to optical_track_predict"<<std::endl;
    }
    ret = eco_track_predict(temp);
    if(!ret)
    {
        std::cerr<<"WARNING: fail to eco_track_predict"<<std::endl;
    }
    
    kf->predict(this->mean, this->covariance);
    this->age += 1;
    this->time_since_update += 1;
}


bool Track::init_optical_track(cv::Mat&frame)
{
    
    optical_track = std::make_shared<MedianFlow>(frame);
    if(optical_track)
    {
        is_init_optical = true;
        return true;
    }
    else
    {
        is_init_optical = false;
        return false;
    }
}
bool Track::init_eco_track(cv::Mat&frame)
{
    eco::EcoParameters paramters;
    eco_track = std::make_shared<eco::ECO>();
    if(eco_track)
    {
        cv::Mat temp = frame.clone();
        eco_track->init(temp,pre_rect,paramters);
        is_init_eco_track = true;
        this->eco_rect = pre_rect;
        return true;
    }
    else
    {
        is_init_eco_track = false;
        return false;
    }
}

bool Track::optical_track_predict(cv::Mat&frame)
{
    if(!is_init_optical)
    {
        std::cerr<<"WARNING:optical_track is not init,fail to predict "<<std::endl;
        return false;
    }
    TYPE_MF_BB rect_;
    rect_.x = pre_rect.x;
    rect_.y = pre_rect.y;
    rect_.height = pre_rect.height;
    rect_.width = pre_rect.width;
    int status = 0;
    cv::Mat temp_frame = frame.clone();
    TYPE_MF_BB temp = optical_track->trackBox(rect_,status,temp_frame);
    if(status != 0)
    {
        optical_rect = cv::Rect(0,0,0,0);
        optical_score = 0.0;
        optical_predict_status = false;
    }
    else
    {
        optical_rect.x = int(temp.x);
        optical_rect.y = int(temp.y);
        optical_rect.height = int(temp.height);
        optical_rect.width = int(temp.width);
        if(optical_rect.area() > 200)
        {
            optical_score = 1.0;
            optical_predict_status = true; 
        }
        else
        {
            optical_rect = cv::Rect(0,0,0,0);
            optical_score = 0.0;
            optical_predict_status = false;
            status = 10000;
        }
    }
    return true;
}



bool Track::eco_track_predict(cv::Mat&frame)
{
    if(!is_init_eco_track)
    {
        std::cerr<<"WARNING:eco_track is not init,fail to predict "<<std::endl;
        return false;
    }
    if(frame.empty())
    {
        std::cout<<"frame is empty"<<std::endl;
        exit(0);
    }
    cv::Rect2f bbox;
    float  confidence = 0.0;
     cv::Mat temp_frame = frame.clone();
    eco_track->update(temp_frame, bbox,confidence);
    if(confidence < 0.15)
    {
        eco_predict_status = false;
        eco_rect.x = 0;
        eco_rect.y = 0;
        eco_rect.height = 0;
        eco_rect.width = 0;
        eco_score = 0.0;
    }
    else
    {
        if(bbox.area() > 200)
        {
            eco_rect.x = int(bbox.x);
            eco_rect.y = int(bbox.y);
            eco_rect.height = int(bbox.height);
            eco_rect.width = int(bbox.width);
            eco_score = confidence;
            eco_predict_status = true;
        }
        else
        {
            eco_predict_status = false;
            eco_rect.x = 0;
            eco_rect.y = 0;
            eco_rect.height = 0;
            eco_rect.width = 0;
            eco_score = 0.0;
        }
    }
    return true;
}



void Track::update(KalmanFilterTrack * const kf, const DETECTION_ROW& detection,bool is_cascade_match)
{
    KAL_DATA pa = kf->update(this->mean, this->covariance, detection.to_xyah());
    this->mean = pa.first;
    this->covariance = pa.second;
    if(is_cascade_match)
    {
        if(detection.is_feature)
        {
            featuresAppendOne(detection.feature);
        }
        
    }
    this->hits += 1;
    this->time_since_update = 0;
    if(this->state == TrackState::Tentative && this->hits >= this->_n_init) 
    {
        this->state = TrackState::Confirmed;
    }
}

void Track::mark_missed()
{
    if(this->state == TrackState::Tentative) {
        this->state = TrackState::Deleted;
    } else if(this->time_since_update > this->_max_age) {
        this->state = TrackState::Deleted;
    }
}
bool Track::set_deleted()
{
    return this->state = TrackState::Deleted;
}

bool Track::is_confirmed()
{
    return this->state == TrackState::Confirmed;
}

bool Track::is_deleted()
{
    return this->state == TrackState::Deleted;
}

bool Track::is_tentative()
{
    return this->state == TrackState::Tentative;
}

DETECTBOX Track::to_tlwh()
{
    DETECTBOX ret = mean.leftCols(4);
    ret(2) *= ret(3);
    ret.leftCols(2) -= (ret.rightCols(2)/2);
    cv::Rect pre_bbox(ret(0), ret(1), ret(2), ret(3));
    predict_box = pre_bbox;
    return ret;
}


DETECTBOX Track::optical_to_tlwh()
{
    DETECTBOX ret = DETECTBOX(optical_rect.x, optical_rect.y,optical_rect.width,  optical_rect.height);
    return ret;
}

DETECTBOX Track::eco_to_tlwh()
{
    DETECTBOX ret = DETECTBOX(eco_rect.x, eco_rect.y,eco_rect.width, eco_rect.height);
    return ret;
}


void Track::featuresAppendOne(const FEATURE &f)
{
    int size = this->features.rows();
    FEATURESS newfeatures = FEATURESS(size+1, 2048);
    newfeatures.block(0, 0, size, 2048) = this->features;
    newfeatures.row(size) = f;
    features = newfeatures;
}
