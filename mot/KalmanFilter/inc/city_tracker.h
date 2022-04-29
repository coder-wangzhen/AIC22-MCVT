
#ifndef _CITY_TRACKER_H
#define _CITY_TRACKER_H
#include <fstream>
#include <sstream>
#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/core/utility.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/core/types_c.h>
#include <opencv2/bgsegm.hpp>
#include "FeatureTensor.h"
#include "tracker.h"
#include "rapidjson/document.h"
#include "rapidjson/writer.h"
#include "rapidjson/stringbuffer.h"
class tracker;
class CityTracker
{
public:
    CityTracker(PARAM_S&param);
    ~CityTracker();
    void load_region(std::string path_json);
    std::vector<std::string> splitString_STD(std::string srcStr, std::string delimStr,bool repeatedCharIgnored);
    double cal_iou(cv::Rect &box1,cv::Rect &box2);
    void draw_region(cv::Mat&frame);
    
    bool init();
    void start();
    void run();

    bool is_crowd_region(int track_idx);
    int calInterMask(cv::Mat&mask1,cv::Mat&mask2);
    void GenerateMask();
    int calMask(cv::Mat&mask);
    int which_region(int track_idx);
    int is_leave(cv::Rect &track_box);
    bool is_available_create(cv::Rect&rect);
    trajectory_status get_traj_status(int track_idx,int&region_id);
    bool is_crowd_region(cv::Rect &detect_box);
    int is_in_leave_region_or_ten(int region_id);
    void load_region_crowd(std::string path_json);
    void GenerateCrowdMask();
    bool is_crowd_region2(int track_idx);
    bool is_contain_crowd_region(cv::Rect &detect_box);
    
public:
    PARAM_S local_param;
    std::string mask_save_path;
    bool m_is_runing;
    bool is_finish;
    int frame_id;
    cv::VideoCapture m_cap;
    std::shared_ptr<tracker> m_tracker_ptr;
    std::thread m_readThread;

    cv::Mat mask;
    cv::Mat unmask;
    cv::Mat enter_mask;
    cv::Mat leave_mask;

    std::map<int,cv::Mat>m_region_mask;
    std::map<int,cv::Mat>m_crowd_region_mask;
    std::map<std::string,std::vector<cv::Point>> m_regions;
    std::map<std::string,std::vector<cv::Point>> m_regions_crowd;

};

#endif
