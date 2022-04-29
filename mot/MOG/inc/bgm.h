#ifndef _SEG_BG_
#define _SEG_BG_


#include "opencv2/core/cuda.hpp"
#include "opencv2/cudabgsegm.hpp"
#include "opencv2/video.hpp"
#include <opencv2/opencv.hpp>
#include "opencv2/opencv_modules.hpp"

#include "common_data.h"
#include<iostream>
#include<error.h>
#include<errno.h>

using namespace cv;
using namespace cv::cuda;

class SegmBg
{

public:
    SegmBg(int init_frame_num,std::string image_path);
    ~SegmBg();

public:

    bool useRoi;

    std::string m_image_path;

    int m_init_frame_num;

    cv::Mat cpu_foreground;

    cv::cuda::GpuMat gpu_fore_ground;

    cv::Mat cup_back_ground;

    cv::cuda::GpuMat gpu_back_ground;

    
    cv::Ptr<cv::BackgroundSubtractorMOG2> mog2;

    RET_INFO init_model(std::string image_path,int m_init_frame_numl);

    RET_INFO get_foreground_mask(cv::Mat &frame,cv::Mat &mask,float learing_rate);

    RET_INFO get_background_image(cv::Mat &back_ground_image);

private:
    int randomInt(int minVal, int maxVal);
    GpuMat createMat(Size size, int type, bool useRoi=false);
    GpuMat loadMat(const Mat& m, bool useRoi);

};


#endif