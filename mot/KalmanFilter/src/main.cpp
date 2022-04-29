#include "city_tracker.h"
#include <condition_variable>


std::string input_video_base_path = "../../datasets/AIC22_Track1_MTMC_Tracking/test/S06/";
std::string feature_file_base_path = "../../datasets/algorithm_results/detect_merge/";
std::string region_base_path = "../region/";
std::string region_crowd_path = "../region_cowd/";
bool is_save_debug_image = false;
//41
std::vector<int> enter_region_41 = {1,3,5,7,10};
std::vector<int> leave_region_41 = {2,4,6,8};
std::vector<int> crowd_region_41 = {5};

//42
std::vector<int> enter_region_42 = {1,3,5,7,10};
std::vector<int> leave_region_42 = {2,4,6,8};
std::vector<int> crowd_region_42 = {5};

//43
std::vector<int> enter_region_43 = {1,3,5,7,10};
std::vector<int> leave_region_43 = {2,4,6,8};
std::vector<int> crowd_region_43 = {5};

//44
std::vector<int> enter_region_44 = {1,3,5,7,10};
std::vector<int> leave_region_44 = {2,4,6,8};
std::vector<int> crowd_region_44 = {3};

//45
std::vector<int> enter_region_45 = {1,3,5,7,10};
std::vector<int> leave_region_45 = {2,4,6,8};
std::vector<int> crowd_region_45 = {3};

//46
std::vector<int> enter_region_46 = {1,3,5,7,10};
std::vector<int> leave_region_46 = {2,4,6,8};
std::vector<int> crowd_region_46 = {5};

int main(int argc, char** argv)
{
    std::vector<std::string> cam_name_list = {"c041","c042","c043","c044","c045","c046"};
    // std::vector<std::string> cam_name_list = {"c042"};
    std::vector<std::shared_ptr<CityTracker>> t_ptr_list;
    for(size_t i = 0; i < cam_name_list.size();i++)
    {
        PARAM_S temp;
        temp.feature_dim = 2048;
        temp.cam_name = cam_name_list[i];
        temp.video_path = input_video_base_path + cam_name_list[i] + "/vdo.avi";
        temp.feature_detect_box_file = feature_file_base_path + cam_name_list[i] + "/feature/";
        temp.region_path = region_base_path + cam_name_list[i] + ".json";
        temp.region_crowd_path = region_crowd_path + cam_name_list[i] + ".json";
        temp.is_save_image = is_save_debug_image;
        std::map<REG_TYPE,std::vector<int>> regions_type;
        std::vector<int> crowd_region;
        if(cam_name_list[i] == "c041")
        {
            temp.min_rect_eare = 600;
            temp.max_age = 50;
            temp.n_init=3;
            temp.nn_budget = 100;
            temp.max_cosine_distance = 2.8;
            temp.merger_distance = 0.4;
            temp.max_iou_distance = 2.8;
            temp.create_score_iou = 0.3;
            temp.create_score = 0.3;
            temp.match_score = 0.1;
            temp.frame_width = 1280;
            temp.frame_height = 960;
            regions_type[ENTER_REGION] = enter_region_41;
            regions_type[LEAVE_REGION] = leave_region_41;
            crowd_region = crowd_region_41;
        }
        if(cam_name_list[i] == "c042")
        {
            temp.max_age = 30;
            temp.min_rect_eare = 600;
            temp.n_init=3;
            temp.nn_budget = 100;
            temp.max_cosine_distance = 2.8;
            temp.merger_distance = 0.4;
            temp.max_iou_distance = 2.8;
            temp.create_score_iou = 0.3;
            temp.create_score = 0.3;
            temp.match_score = 0.1;
            temp.frame_width = 1280;
            temp.frame_height = 960;
            regions_type[ENTER_REGION] = enter_region_42;
            regions_type[LEAVE_REGION] = leave_region_42;
            crowd_region = crowd_region_42;
        }
        if(cam_name_list[i] == "c043")
        {
            temp.max_age = 30;
            temp.min_rect_eare = 600;
            temp.n_init=3;
            temp.nn_budget = 100;
            temp.max_cosine_distance = 2.8;
            temp.merger_distance = 0.4;
            temp.max_iou_distance = 2.8;
            temp.create_score_iou = 0.3;
            temp.create_score = 0.3;
            temp.match_score = 0.1;
            temp.frame_width = 1280;
            temp.frame_height = 960;
            regions_type[ENTER_REGION] = enter_region_43;
            regions_type[LEAVE_REGION] = leave_region_43;
            crowd_region = crowd_region_43;
        }
        if(cam_name_list[i] == "c044")
        {
            temp.max_age = 30;
            temp.min_rect_eare = 600;
            temp.n_init=3;
            temp.nn_budget = 100;
            temp.max_cosine_distance = 2.8;
            temp.merger_distance = 0.4;
            temp.max_iou_distance = 2.8;
            temp.create_score_iou = 0.3;
            temp.create_score = 0.3;
            temp.match_score = 0.1;
            temp.frame_width = 1280;
            temp.frame_height = 960;
            regions_type[ENTER_REGION] = enter_region_44;
            regions_type[LEAVE_REGION] = leave_region_44;
            crowd_region = crowd_region_44;
        }
        if(cam_name_list[i] == "c045")
        {
            temp.max_age = 30;
            temp.min_rect_eare = 600;
            temp.n_init=3;
            temp.nn_budget = 100;
            temp.max_cosine_distance = 2.8;
            temp.merger_distance = 0.4;
            temp.max_iou_distance = 2.8;
            temp.create_score_iou = 0.3;
            temp.create_score = 0.3;
            temp.match_score = 0.1;
            temp.frame_width = 1280;
            temp.frame_height = 720;
            regions_type[ENTER_REGION] = enter_region_45;
            regions_type[LEAVE_REGION] = leave_region_45;
            crowd_region = crowd_region_45;
        }
        if(cam_name_list[i] == "c046")
        {
            temp.max_age = 30;
            temp.min_rect_eare = 600;
            temp.n_init=3;
            temp.nn_budget = 100;
            temp.max_cosine_distance = 2.8;
            temp.merger_distance = 0.4;
            temp.max_iou_distance = 2.8;
            temp.create_score_iou = 0.3;
            temp.create_score = 0.3;
            temp.match_score = 0.1;
            temp.frame_width = 1280;
            temp.frame_height = 720;
            regions_type[ENTER_REGION] = enter_region_46;
            regions_type[LEAVE_REGION] = leave_region_46;
            crowd_region = crowd_region_46;
        }
        temp.regions_type = regions_type;
        temp.crowd_regions = crowd_region;
        std::shared_ptr<CityTracker> ptr = std::make_shared<CityTracker>(temp);
        ptr->init();
        ptr->start();
        t_ptr_list.push_back(ptr);
    }
    while(1)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(5000));
        bool is_finish = true;
        for(size_t i = 0; i < t_ptr_list.size();i++)
        {
            if(!t_ptr_list[i]->is_finish)
            {
                is_finish = false;
                break;
            }
        }
        if(is_finish)
        {
            std::cout<<"*****************save file*************"<<std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(300000));
            t_ptr_list.clear();
            std::cout<<"*****************finish*************"<<std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(6000));
            exit(0);
        }
    }
    
    return 0;
}

