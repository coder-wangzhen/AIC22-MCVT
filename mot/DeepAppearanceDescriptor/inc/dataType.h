#pragma once
#ifndef DATATYPE_H
#define DATATYPE_H

#include <cstddef>
#include <vector>
#include <map>
#include<iostream>
#include<vector>
#include<string>
#include <Eigen/Core>
#include <Eigen/Dense>
typedef Eigen::Matrix<float, 1, 4, Eigen::RowMajor> DETECTBOX;
typedef Eigen::Matrix<float, -1, 4, Eigen::RowMajor> DETECTBOXSS;
typedef Eigen::Matrix<float, 1, 2048, Eigen::RowMajor> FEATURE;
typedef Eigen::Matrix<float, Eigen::Dynamic, 2048, Eigen::RowMajor> FEATURESS;
//typedef std::vector<FEATURE> FEATURESS;

//Kalmanfilter
//typedef Eigen::Matrix<float, 8, 8, Eigen::RowMajor> KAL_FILTER;
typedef Eigen::Matrix<float, 1, 8, Eigen::RowMajor> KAL_MEAN;
typedef Eigen::Matrix<float, 8, 8, Eigen::RowMajor> KAL_COVA;
typedef Eigen::Matrix<float, 1, 4, Eigen::RowMajor> KAL_HMEAN;
typedef Eigen::Matrix<float, 4, 4, Eigen::RowMajor> KAL_HCOVA;
using KAL_DATA = std::pair<KAL_MEAN, KAL_COVA>;
using KAL_HDATA = std::pair<KAL_HMEAN, KAL_HCOVA>;

//main
using RESULT_DATA = std::pair<int, DETECTBOX>;

//tracker:
using TRACKER_DATA = std::pair<int, FEATURESS>;
using MATCH_DATA = std::pair<int, int>;

typedef struct t{
    std::vector<MATCH_DATA> matches;
    std::vector<int> unmatched_tracks;
    std::vector<int> unmatched_detections;
}TRACHER_MATCHD;

//linear_assignment:
typedef Eigen::Matrix<float, -1, -1, Eigen::RowMajor> DYNAMICM;


enum REG_TYPE
{
    OUT_SIDE_REGION,
    ENTER_REGION,
    LEAVE_REGION
};
typedef struct param
{
    std::string cam_name;
    std::string video_path;
    std::string high_threshold_detect_result;
    std::string low_threshold_detect_result;
    std::string feature_detect_box_file;
    std::string region_path;
    int nn_budget;
    float max_cosine_distance;
    float merger_distance;
    float max_iou_distance;
    float match_score;
    int min_rect_eare;
    float create_score;
    float create_score_iou;
    int feature_dim;
    int frame_width;
    int frame_height;
    int max_age;
    int n_init;
    std::map<REG_TYPE,std::vector<int>> regions_type;
    std::vector<int> crowd_regions;
    
    std::string mask_save_path;
    std::string region_crowd_path;
    bool is_save_image;
    param()
    {
        is_save_image = false;
        mask_save_path = "../mask/";
        cam_name = "test";
        nn_budget = 100;
        max_cosine_distance = 2.6;
        max_iou_distance = 2.6;
        feature_dim = 2048;
        merger_distance = 0.7;
        match_score = 0.1;
        create_score = 0.4;
        create_score_iou = 0.2;
        max_age = 500;
        n_init = 3;
        min_rect_eare = 1000;
    };

}PARAM_S;


#endif // DATATYPE_H
