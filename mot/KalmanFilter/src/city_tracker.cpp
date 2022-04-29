#include "city_tracker.h"

CityTracker::CityTracker(PARAM_S&param)
{
    local_param.cam_name = param.cam_name;
    local_param.video_path = param.video_path;
    local_param.feature_detect_box_file = param.feature_detect_box_file;
    local_param.region_path = param.region_path;
    local_param.nn_budget = param.nn_budget;
    local_param.max_cosine_distance = param.max_cosine_distance;
    local_param.merger_distance = param.merger_distance;
    local_param.max_iou_distance = param.max_iou_distance;
    local_param.match_score = param.match_score;
    local_param.min_rect_eare = param.min_rect_eare;
    local_param.create_score = param.create_score;
    local_param.create_score_iou = param.create_score_iou;
    local_param.feature_dim = param.feature_dim;
    local_param.frame_width = param.frame_width;
    local_param.frame_height = param.frame_height;
    local_param.max_age = param.max_age;
    local_param.n_init = param.n_init;
    local_param.regions_type = param.regions_type;
    local_param.crowd_regions = param.crowd_regions;
    local_param.mask_save_path = param.mask_save_path;
    local_param.region_crowd_path = param.region_crowd_path;
    local_param.is_save_image = param.is_save_image;
    frame_id = -1;
    m_is_runing = false;
    is_finish = false;
    mask_save_path = local_param.mask_save_path + local_param.cam_name + "/";
}


CityTracker::~CityTracker()
{
    m_is_runing = false;
    if (m_readThread.joinable())
    {
        m_readThread.join();
    }
}

bool CityTracker::init()
{
    
    load_region(local_param.region_path);
    load_region_crowd(local_param.region_crowd_path);
    GenerateCrowdMask();
    GenerateMask();
    return true;
}
void CityTracker::GenerateCrowdMask()
{
    cv::Mat crowd_region = cv::Mat(local_param.frame_height,local_param.frame_width,
    CV_8UC1,cv::Scalar(0));
   

    std::map<std::string,std::vector<cv::Point>>::iterator iter_r;
    for(iter_r = m_regions_crowd.begin(); iter_r != m_regions_crowd.end();iter_r++)
    {
        int region_id = std::atoi(iter_r->first.c_str());
        cv::Mat temp = cv::Mat(local_param.frame_height,local_param.frame_width,
        CV_8UC1,cv::Scalar(0));
        std::vector<cv::Point> region_temp = iter_r->second;
        cv::fillPoly(temp, region_temp, Scalar(255), 8, 0);
        m_crowd_region_mask[region_id] = temp;
        // cv::imshow(local_param.cam_name+"-label-"+iter_r->first,temp);
        // cv::waitKey(5000);
    }

}

void CityTracker::start()
{
    m_is_runing=true;
    m_readThread = std::thread(std::bind(&CityTracker::run, this));
}

void CityTracker::run()
{
    cv::VideoCapture m_cap;

    m_cap.open(local_param.video_path.c_str());

    if(!m_cap.isOpened())
    {
        std::cerr<<"ERROR:Fail to open video: "<<local_param.video_path<<std::endl;
        return;
    }
    
    m_tracker_ptr = std::make_shared<tracker>(this,local_param.max_cosine_distance,
    local_param.merger_distance,local_param.nn_budget,local_param.max_iou_distance,
    local_param.max_age,local_param.n_init);
    
    int frame_id = -1;
    while(m_is_runing)
    {
        if(is_finish)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(5000));
            continue;
        }
        cv::Mat frame;
        bool ret = m_cap.read(frame);
        if(!ret)
        {
            is_finish = true;
            std::cout<<"Fail to read frame from video: "<<local_param.video_path<<std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(5000));
            continue;
        }
        frame_id++;
        //std::cout<<"frame_id:"<<frame_id<<std::endl;
        DETECTIONS match_detect_b;
        DETECTIONS create_detect_box;
        FeatureTensor::getInstance()->getRectsFeature(match_detect_b,create_detect_box,frame_id,
         local_param);
        std::cout<<local_param.cam_name +"***************************frame_id: "<<frame_id<<std::endl;
        m_tracker_ptr->predict(frame);
        m_tracker_ptr->update(match_detect_b,create_detect_box,frame,frame_id);
        
       
        // cv::imshow(local_param.cam_name.c_str(),frame);
        
        // cv::waitKey(1);
    }
}


void CityTracker::draw_region(cv::Mat&frame)
{
    std::map<std::string,std::vector<cv::Point>>::iterator iter;
    for(iter = m_regions.begin(); iter != m_regions.end();iter++)
    {   
        cv::Point pt = iter->second[3];
        if(iter->first == "1")
        {
            cv::polylines(frame, iter->second, true, Scalar(0, 0, 255), 3, cv::LINE_AA);
            cv::putText(frame, iter->first, pt, 
            cv::FONT_HERSHEY_SIMPLEX, 1.6, Scalar(0, 0, 255), 1);
            
        }
        if(iter->first == "2")
        {
            cv::polylines(frame, iter->second, true, Scalar(255, 0, 255), 3, cv::LINE_AA);
            cv::putText(frame, iter->first, pt, 
            cv::FONT_HERSHEY_SIMPLEX, 1.6, Scalar(255, 0, 255), 1);
            
        }
        if(iter->first == "3")
        {
            cv::polylines(frame, iter->second, true, Scalar(0, 255, 255), 3, cv::LINE_AA);
            cv::putText(frame, iter->first, pt, 
            cv::FONT_HERSHEY_SIMPLEX, 1.6, Scalar(0, 255, 255), 1);
            
        }
        if(iter->first == "4")
        {
            cv::polylines(frame, iter->second, true, Scalar(255, 0, 0), 3, cv::LINE_AA);
            cv::putText(frame, iter->first, pt, 
            cv::FONT_HERSHEY_SIMPLEX, 1.6, Scalar(255, 0, 0), 1);
            
        }
        if(iter->first == "5")
        {
            cv::polylines(frame, iter->second, true, Scalar(0, 255, 0), 3, cv::LINE_AA);
            cv::putText(frame, iter->first, pt, 
            cv::FONT_HERSHEY_SIMPLEX, 1.6, Scalar(0, 255, 0), 1);
            
        }
        if(iter->first == "6")
        {
            cv::polylines(frame, iter->second, true, Scalar(255, 255, 0), 3, cv::LINE_AA);
            cv::putText(frame, iter->first, pt, 
            cv::FONT_HERSHEY_SIMPLEX, 1.6, Scalar(255, 255, 0), 1);
            
        }
        if(iter->first == "7")
        {
            cv::polylines(frame, iter->second, true, Scalar(0, 0, 255), 3, cv::LINE_AA);
            cv::putText(frame, iter->first, pt, 
            cv::FONT_HERSHEY_SIMPLEX, 1.6, Scalar(0, 0, 255), 1);
            
        }
        if(iter->first == "8")
        {
            cv::polylines(frame, iter->second, true, Scalar(150, 150, 255), 3, cv::LINE_AA);
            cv::putText(frame, iter->first, pt, 
            cv::FONT_HERSHEY_SIMPLEX, 1.6, Scalar(150, 150, 255), 1);
            
        }
        if(iter->first == "10")
        {
            cv::polylines(frame, iter->second, true, Scalar(90, 255, 150), 3, cv::LINE_AA);
            cv::putText(frame, iter->first, pt, 
            cv::FONT_HERSHEY_SIMPLEX, 1.6, Scalar(90, 255, 150), 1);
            
        }
    }
}


void CityTracker::load_region_crowd(std::string path_json)
{
    std::ifstream t(path_json.c_str());
    std::string str((std::istreambuf_iterator<char>(t)),
                  std::istreambuf_iterator<char>());
    rapidjson::Document document;
    if(str.empty())
	{
		std::cerr<<"file is empty:"<<str<<std::endl;
		return;
	}
   
    document.Parse(str.c_str());
    if (document.HasParseError())
	{
		printf("parse失败:%d\n", document.GetParseError());
	}
    else
    {
        std::cout<<"succeed to Parse document"<<std::endl;
    }

    std::cerr<<"load_region_crowd path_json: "<<path_json<<std::endl;

    rapidjson::Value::ConstMemberIterator iter = document.FindMember("shapes");
    if(iter != document.MemberEnd())
    //if(document.HasMember("shapes"))
    {
        
        if(iter->value.IsArray() == false)
        {
            std::cerr<<"document[shapes] is not array"<<std::endl;
        }
        
        const rapidjson::Value& array = iter->value;
        if(array.IsArray() == false)
        {
            std::cerr<<"document[shapes] is not array"<<std::endl;
        }
        for(size_t i = 0; i < array.Size(); i++)
        {
           
            const rapidjson::Value& chileValue = array[i];
            std::vector<cv::Point> label_region;
            std::string label = std::string(chileValue["label"].GetString());
            
            const rapidjson::Value& points_list = chileValue["points"];
            for(size_t j = 0; j < points_list.Size(); j++)
            {
                const rapidjson::Value& point = points_list[j];
                if(point.IsArray() == false)
                {
                    std::cout<<"point is not array"<<std::endl;
                    continue;
                }
                if(point.Size() == 2)
                {
                    int x = int(point[0].GetFloat());
                    int y = int(point[1].GetFloat());
                    cv::Point pt = cv::Point(x,y);
                    label_region.push_back(pt);
                }
            }
            m_regions_crowd[label] = label_region;  
        }
    }
    else
    {
        std::cout<<"document do not contain  shapes"<<std::endl;
    }
    std::cout<<"m_regions_crowd size:"<<m_regions_crowd.size()<<std::endl;
    if(m_regions_crowd.size() == 0)
    {
        std::cout<<"fail to load json"<<std::endl;
        exit(0);
    }
}

void CityTracker::load_region(std::string path_json)
{
    std::ifstream t(path_json.c_str());
    std::string str((std::istreambuf_iterator<char>(t)),
                  std::istreambuf_iterator<char>());
    rapidjson::Document document;
    if(str.empty())
	{
		std::cerr<<"file is empty:"<<str<<std::endl;
		return;
	}
   
    document.Parse(str.c_str());
    if (document.HasParseError())
	{
		printf("parse失败:%d\n", document.GetParseError());
	}
    else
    {
        std::cout<<"succeed to Parse document"<<std::endl;
    }

    std::cerr<<"region path_json: "<<path_json<<std::endl;

    rapidjson::Value::ConstMemberIterator iter = document.FindMember("shapes");
    if(iter != document.MemberEnd())
    //if(document.HasMember("shapes"))
    {
        
        if(iter->value.IsArray() == false)
        {
            std::cerr<<"document[shapes] is not array"<<std::endl;
        }
        
        const rapidjson::Value& array = iter->value;
        if(array.IsArray() == false)
        {
            std::cerr<<"document[shapes] is not array"<<std::endl;
        }
        for(size_t i = 0; i < array.Size(); i++)
        {
           
            const rapidjson::Value& chileValue = array[i];
            std::vector<cv::Point> label_region;
            std::string label = std::string(chileValue["label"].GetString());
            
            const rapidjson::Value& points_list = chileValue["points"];
            for(size_t j = 0; j < points_list.Size(); j++)
            {
                const rapidjson::Value& point = points_list[j];
                if(point.IsArray() == false)
                {
                    std::cout<<"point is not array"<<std::endl;
                    continue;
                }
                if(point.Size() == 2)
                {
                    int x = int(point[0].GetFloat());
                    int y = int(point[1].GetFloat());
                    cv::Point pt = cv::Point(x,y);
                    label_region.push_back(pt);
                }
            }
            m_regions[label] = label_region;  
        }
    }
    else
    {
        std::cout<<"document do not contain  shapes"<<std::endl;
    }
}


int CityTracker::calInterMask(cv::Mat&mask1,cv::Mat&mask2)
{
    if(mask1.empty())
    {
        std::cerr<<"mask1 is empty"<<std::endl;
        exit(0);
    }
    if(mask2.empty())
    {
        std::cerr<<"mask1 is empty"<<std::endl;
        exit(0);
    }
    cv::Mat intersection = mask1 & mask2;
    return calMask(intersection);
}

int CityTracker::calMask(cv::Mat&mask)
{
    int area = 0;
    for(int i=0;i<mask.rows;++i)
    {
        for(int j=0;j<mask.cols;++j)
        {
            if(mask.at<uchar>(i,j) > 200)
            {
                area++;
            } 
        }
    }
    return area;
}


bool CityTracker::is_available_create(cv::Rect&rect)
{
    cv::Mat vechile_r = cv::Mat (local_param.frame_height,local_param.frame_width,CV_8UC1,Scalar(0));
    cv::rectangle(vechile_r, rect, Scalar(255), -1);
    int inter_leave_region_area = calInterMask(vechile_r,leave_mask);
    int inter_out_region_area = calInterMask(vechile_r,unmask);
    int self_region_area = calMask(vechile_r);
    if((inter_out_region_area*1.0 / self_region_area*1.0) > 0.9)
    {
        return false;
    }
    if((inter_leave_region_area > 20)&&(inter_out_region_area > 20))
    {
        return false;
    }
    return true;
}

int CityTracker::which_region(int track_idx)
{
    cv::Rect rect = m_tracker_ptr->tracks[track_idx].track_box;

    if(m_tracker_ptr->tracks[track_idx].track_infomation->traj_status == INIT)
    {
        int reg_id = -1;
        for(size_t i = 0; i < local_param.regions_type[ENTER_REGION].size();i++)
        {
            reg_id = local_param.regions_type[ENTER_REGION][i];
            cv::Mat temp = m_region_mask[reg_id];
            cv::Mat vechile_r = cv::Mat (local_param.frame_height,local_param.frame_width,CV_8UC1,Scalar(0));
            cv::rectangle(vechile_r, rect, Scalar(255), -1);
           
            if(calInterMask(temp,vechile_r) > 0)
            {
                return reg_id;
            }
        }

        int x = rect.x + rect.width/2;
        int y = rect.y + rect.height/2;
        cv::Point center_point = cv::Point(x,y);
    
        std::map<std::string,std::vector<cv::Point>>::iterator iter;
        for(iter = m_regions.begin();iter != m_regions.end();iter++)
        {
            double ret = pointPolygonTest(iter->second, center_point, false);
            if(ret < 0)
            {
                continue;
            }
            else
            {
                return std::atoi(iter->first.c_str());
            }

        }
        cv::Mat vechile_r = cv::Mat (local_param.frame_height,local_param.frame_width,CV_8UC1,Scalar(0));
        cv::rectangle(vechile_r, rect, Scalar(255), -1);
        std::map<int,cv::Mat>::iterator iter_;
        for(iter_ = m_region_mask.begin();iter_ != m_region_mask.end();iter_++)
        {
            cv::Mat inter_ = vechile_r & iter_->second;
            if(calMask(inter_) > 0)
            {
                return iter_->first;
            }
        }

        std::cerr<<"current_frame_id: "<<m_tracker_ptr->current_frame_id<<std::endl;
        std::cerr<<local_param.cam_name+"-INIT ERROR: "<<m_tracker_ptr->tracks[track_idx].track_id<<std::endl;
        exit(0);
    }
    

    else if(m_tracker_ptr->tracks[track_idx].track_infomation->traj_status == ENTER)
    {
        int x = rect.x + rect.width/2;
        int y = rect.y + rect.height/2;
        cv::Point center_point = cv::Point(x,y);
    
        std::map<std::string,std::vector<cv::Point>>::iterator iter;
        for(iter = m_regions.begin();iter != m_regions.end();iter++)
        {
            double ret = pointPolygonTest(iter->second, center_point, false);
            if(ret < 0)
            {
                continue;
            }
            else
            {
                return std::atoi(iter->first.c_str());
            }

        }
        cv::Mat vechile_r = cv::Mat (local_param.frame_height,local_param.frame_width,CV_8UC1,Scalar(0));
        cv::rectangle(vechile_r, rect, Scalar(255), -1);
        std::map<int,cv::Mat>::iterator iter_;
        for(iter_ = m_region_mask.begin();iter_ != m_region_mask.end();iter_++)
        {
            cv::Mat inter_ = vechile_r & iter_->second;
            if(calMask(inter_) > 0)
            {
                return iter_->first;
            }
        }
        std::cerr<<"current_frame_id: "<<m_tracker_ptr->current_frame_id<<std::endl;
        std::cerr<<local_param.cam_name+"ENTER ERROR: "<<m_tracker_ptr->tracks[track_idx].track_id<<std::endl;
        exit(0);
    }
    else if(m_tracker_ptr->tracks[track_idx].track_infomation->traj_status == READY_LEAVE)
    {
        int reg_id = -1;
        for(size_t i = 0; i < local_param.regions_type[LEAVE_REGION].size();i++)
        {
            reg_id = local_param.regions_type[LEAVE_REGION][i];
            cv::Mat temp = m_region_mask[reg_id];
            cv::Mat vechile_r = cv::Mat (local_param.frame_height,local_param.frame_width,CV_8UC1,Scalar(0));
            cv::rectangle(vechile_r, rect, Scalar(255), -1);
            if(calInterMask(temp,vechile_r) > 0)
            {
                return reg_id;
            }
        }
        std::cerr<<"current_frame_id: "<<m_tracker_ptr->current_frame_id<<std::endl;
        std::cerr<<local_param.cam_name+"READY_LEAVE ERROR: "<<m_tracker_ptr->tracks[track_idx].track_id<<std::endl;
        exit(0);
        // if(reg_id == -1)
        // {
        //     std::cerr<<"READY_LEAVE ERROR"<<std::endl;
        //     exit(0);
        // }
    }
    
    else if(m_tracker_ptr->tracks[track_idx].track_infomation->traj_status == LEAVE)
    {
        int reg_id = -1;
        for(size_t i = 0; i < local_param.regions_type[LEAVE_REGION].size();i++)
        {
            reg_id = local_param.regions_type[LEAVE_REGION][i];
            cv::Mat temp = m_region_mask[reg_id];
            cv::Mat vechile_r = cv::Mat (local_param.frame_height,local_param.frame_width,CV_8UC1,Scalar(0));
            cv::rectangle(vechile_r, rect, Scalar(255), -1);
            // if(m_tracker_ptr->tracks[track_idx].track_id == 16)
            // {
            //     std::cout<<"leave region id :"<<reg_id<<std::endl;
            //     if(reg_id == 8)
            //     {
            //         std::cout<<"calInterMask(temp,vechile_r): "<<calInterMask(temp,vechile_r)<<std::endl;
            //         cv::imshow("mask",temp);
            //         cv::imshow("vechile_r",vechile_r);
            //         cv::Mat temp_oo = temp & vechile_r;
            //         cv::imshow("temp_oo",temp_oo);
            //         cv::waitKey(500000);
            //     }
            // }
            if(calInterMask(temp,vechile_r) > 0)
            {
                
                return reg_id;
            }
        }
        return -1;
        // if(reg_id == -1)
        // {
        //     std::cerr<<"READY_LEAVE ERROR"<<std::endl;
        //     exit(0);
        // }
    } 
    else
    {
        std::cerr<<"current_frame_id: "<<m_tracker_ptr->current_frame_id<<std::endl;
        std::cerr<<local_param.cam_name+"m_tracker_ptr->tracks[track_idx].track_infomation->traj_status ERROR"<<std::endl;
        exit(0);
        return -1;
    }   
    
    
}

int CityTracker::is_leave(cv::Rect &track_box)   //1 rl 2 l 0 en
{
    cv::Mat vechile_r = cv::Mat (local_param.frame_height,local_param.frame_width,CV_8UC1,Scalar(0));
    cv::rectangle(vechile_r, track_box, Scalar(255), -1);
    int inter_leave_region_area = calInterMask(vechile_r,leave_mask);
    int inter_out_region_area = calInterMask(vechile_r,unmask);
    int self_region_area = calMask(vechile_r);

    if((inter_out_region_area*1.0 / self_region_area*1.0) > 0.9)
    {
        return 2;
    }
    if((inter_leave_region_area > 1)&&(inter_out_region_area > 1))
    {
        return 1;
    }
    return 0;
}   

int CityTracker::is_in_leave_region_or_ten(int region_id)
{
    std::vector<int>::iterator iter = std::find(local_param.regions_type[LEAVE_REGION].begin(),
    local_param.regions_type[LEAVE_REGION].end(),region_id);
    if(iter != local_param.regions_type[LEAVE_REGION].end())
    {
        return 1;
    }
    if(region_id == 10)
    {
        return 2;
    }
    return 0;
}
trajectory_status CityTracker::get_traj_status(int track_idx,int&region_id)
{
    region_id = -1;
    if(m_tracker_ptr->tracks[track_idx].track_infomation->traj_status == INIT)
    {
        
        region_id = which_region(track_idx);

        if(region_id == -1)
        {
            std::cerr<<"error region id"<<std::endl;
            exit(0);
        }

        
        m_tracker_ptr->tracks[track_idx].track_infomation->traj_status = ENTER;
        
        return ENTER;
    } 
    if(m_tracker_ptr->tracks[track_idx].track_infomation->traj_status == ENTER)
    {
        int status_index = is_leave(m_tracker_ptr->tracks[track_idx].track_box);

        if(status_index == 0)
        {
            
            m_tracker_ptr->tracks[track_idx].track_infomation->traj_status = ENTER;
            region_id = which_region(track_idx);
            if(region_id == -1)
            {
                std::cerr<<" ENTER error region id"<<std::endl;
                exit(0);
            }
            return ENTER;
        }
        else if(status_index == 1)
        {
            
            m_tracker_ptr->tracks[track_idx].track_infomation->ready_l++;
            m_tracker_ptr->tracks[track_idx].track_infomation->traj_status = READY_LEAVE;
            region_id = which_region(track_idx);
            if(region_id == -1)
            {
                std::cerr<<" READY_LEAVE error region id"<<std::endl;
                exit(0);
            }
            return READY_LEAVE;
        }
        else if(status_index == 2)
        {
            m_tracker_ptr->tracks[track_idx].track_infomation->traj_status = LEAVE;
            region_id = which_region(track_idx);
            
            return LEAVE;
        }
        else
        {
            std::cerr<<"error status_index"<<std::endl;
            exit(0);
        }
    }

    if(m_tracker_ptr->tracks[track_idx].track_infomation->traj_status == READY_LEAVE)
    {
        

        int status_index = is_leave(m_tracker_ptr->tracks[track_idx].track_box);

        if(status_index == 0)
        {
            m_tracker_ptr->tracks[track_idx].track_infomation->traj_status = ENTER;
            region_id = which_region(track_idx);
            if(region_id == -1)
            {
                std::cerr<<"* ENTER error region id"<<std::endl;
                exit(0);
            }
            return ENTER;
        }
        else if(status_index == 1)
        {
            m_tracker_ptr->tracks[track_idx].track_infomation->ready_l++;
            if(m_tracker_ptr->tracks[track_idx].track_infomation->ready_l > 4)
            {
                m_tracker_ptr->tracks[track_idx].track_infomation->traj_status = LEAVE;
                region_id = which_region(track_idx);
                if(region_id == -1)
                {
                    std::cerr<<"* LEAVE error region id"<<std::endl;
                    exit(0);
                }
                return LEAVE;
            }
            m_tracker_ptr->tracks[track_idx].track_infomation->traj_status = READY_LEAVE;
            region_id = which_region(track_idx);
            if(region_id == -1)
            {
                std::cerr<<"* READY_LEAVE error region id"<<std::endl;
                exit(0);
            }
            return READY_LEAVE; 
        }
        else if(status_index == 2)
        {
            m_tracker_ptr->tracks[track_idx].track_infomation->traj_status = LEAVE;
            region_id = which_region(track_idx);
            
            return LEAVE;
        }
        else
        {
            std::cerr<<"error status_index"<<std::endl;
            exit(0);
        }
    }
    if(m_tracker_ptr->tracks[track_idx].track_infomation->traj_status == LEAVE)
    {
        m_tracker_ptr->tracks[track_idx].track_infomation->traj_status = LEAVE;
        region_id = which_region(track_idx);
        
        return LEAVE;
    }
    else
    {
        std::cerr<<"ERROR**************************************"<<std::endl;
        exit(0);
    }     
}

void CityTracker::GenerateMask()
{
    cv::Mat leave_region = cv::Mat(local_param.frame_height,local_param.frame_width,
    CV_8UC1,cv::Scalar(0));
    cv::Mat enter_region = cv::Mat(local_param.frame_height,local_param.frame_width,
    CV_8UC1,cv::Scalar(0));

    std::map<REG_TYPE,std::vector<int>>::iterator iter;

    for(iter = local_param.regions_type.begin();iter != local_param.regions_type.end();iter++)
    {
        if(iter->first == ENTER_REGION)
        {
            for(size_t i = 0; i < iter->second.size();i++)
            {
                std::vector<cv::Point> region = m_regions[std::to_string(iter->second[i])];
                cv::fillPoly(enter_region, region, Scalar(255), 8, 0);
            }
            
        }
        if(iter->first == LEAVE_REGION)
        {
            for(size_t i = 0; i < iter->second.size();i++)
            {
                std::vector<cv::Point> region = m_regions[std::to_string(iter->second[i])];
                cv::fillPoly(leave_region, region, Scalar(255), 8, 0);
            }
        }
    }


    
    std::map<std::string,std::vector<cv::Point>>::iterator iter_r;
    for(iter_r = m_regions.begin(); iter_r != m_regions.end();iter_r++)
    {
        int region_id = std::atoi(iter_r->first.c_str());
        cv::Mat temp = cv::Mat(local_param.frame_height,local_param.frame_width,
        CV_8UC1,cv::Scalar(0));
        std::vector<cv::Point> region_temp = iter_r->second;
        cv::fillPoly(temp, region_temp, Scalar(255), 8, 0);
        m_region_mask[region_id] = temp;
        std::string save_path = mask_save_path + iter_r->first + ".jpg";
        cv::imwrite(save_path,temp);
        //cv::imshow("label-"+iter_r->first,temp);
    }

    
    cv::Mat element_ = getStructuringElement(MORPH_RECT,
    Size(3, 3));
	
	morphologyEx(enter_region, enter_region, MORPH_CLOSE,element_);

    mask = leave_region | enter_region;

    std::string save_path = mask_save_path + "mask.jpg";
    std::cout<<"*************save_path:"<<save_path<<std::endl;
    cv::imwrite(save_path,mask);

    Mat element = getStructuringElement(MORPH_RECT,
    Size(3, 3));
	
	morphologyEx(mask, mask, MORPH_CLOSE,element);

    unmask = ~mask;

    save_path = mask_save_path + "unmask.jpg";
    cv::imwrite(save_path,unmask);

    // vector<vector<Point>> contours;
	// vector<Vec4i> hierarchy;
	// cv::findContours(unmask,contours,hierarchy,RETR_TREE,CHAIN_APPROX_SIMPLE,Point());
    // //drawContours(unmask, contours, -1, Scalar(128), -1);
    // int leave = contours.size();
    // if(leave != 1)
    // {
    //     std::cout<<"leave region coutours: "<<contours.size()<<std::endl;
        
    //     for (size_t i = 0; i < contours.size(); i++) 
    //     {
    //         std::cout<<"area coutours: "<<cv::contourArea(contours[i])<<std::endl;
    //     }   
        
    //     drawContours(unmask, contours, -1, Scalar(128), 5);
        
    //     cv::imshow("unmask:" +local_param.cam_name,unmask);
    //     cv::waitKey(5000);
    //     Mat element = getStructuringElement(MORPH_RECT,
    //     Size(3, 3));
        
    //     morphologyEx(unmask, unmask, MORPH_OPEN,element);
    //     cv::findContours(unmask,contours,hierarchy,RETR_TREE,CHAIN_APPROX_SIMPLE,Point());
    //     leave = contours.size();

        
    // }
    // std::cout<<"leave region coutours: "<<contours.size()<<std::endl;
    // cv::imshow("unmask",unmask);


    //if(leave)
    // std::cout<<"leave region coutours: "<<contours.size()<<std::endl;
    // for (size_t i = 0; i < contours.size(); i++) 
    // {

    // }   
    // contourArea()

    enter_mask = enter_region.clone();
    leave_mask = leave_region.clone();

    save_path = mask_save_path + "enter.jpg";
    cv::imwrite(save_path,enter_mask);

    save_path = mask_save_path + "leave.jpg";
    cv::imwrite(save_path,leave_mask);

    // cv::imshow("mask",mask);
    // cv::imshow("unmask",unmask);
    // cv::imshow("enter_mask",enter_mask);
    // cv::imshow("leave_mask",leave_mask);
    // cv::waitKey(500000);



    
}


double CityTracker::cal_iou(cv::Rect &box1,cv::Rect &box2)
{
    cv::Rect rect_union = box1 | box2;

    if(rect_union.area() == 0)
    {
        std::cout<<"WARNING:rect_union.area()==0"<<std::endl;
        return 0.0;
    }
    cv::Rect intersetion = box1 & box2;
    if(intersetion.area() == 0)
    {
        //std::cout<<"WARNING:intersetion.area() = 0;"<<std::endl;
      return 0.0;
    }
    double IOU = intersetion.area() *1.0/ rect_union.area()*1.0;
    return IOU;
}




bool CityTracker::is_crowd_region(int track_idx)
{
    int region_id =  which_region(track_idx);
    std::vector<int>::iterator result = std::find(local_param.crowd_regions.begin(),
    local_param.crowd_regions.end(),region_id);
    if(result != local_param.crowd_regions.end())
    {
        return true;
    }
    else
    {
        return false;
    }
}   


bool CityTracker::is_crowd_region(cv::Rect &detect_box)
{
    cv::Mat crowd_region = cv::Mat(local_param.frame_height,local_param.frame_width,
    CV_8UC1,cv::Scalar(0));
    for(size_t j = 0; j < local_param.crowd_regions.size();j++)
    {
        std::vector<cv::Point> region = m_regions[std::to_string(local_param.crowd_regions[j])];
        cv::fillPoly(crowd_region, region, Scalar(255), 8, 0);
    }
    cv::Mat vechile_r = cv::Mat (local_param.frame_height,local_param.frame_width,CV_8UC1,Scalar(0));
    cv::rectangle(vechile_r, detect_box, Scalar(255), -1);
    int inter_crowd_region_area = calInterMask(vechile_r,crowd_region);
    if(inter_crowd_region_area > 15)
    {
        return true;
    }
    else
    {
        return false;
    } 
} 



bool CityTracker::is_crowd_region2(int track_idx)
{
    cv::Mat crowd_region = m_region_mask[5];
    cv::Rect rect = m_tracker_ptr->tracks[track_idx].track_box;
    if(rect.area() == 0)
    {
        std::cout<<"rect is empty"<<std::endl;
        exit(0);
    }
    cv::Mat vechile_r = cv::Mat (local_param.frame_height,local_param.frame_width,CV_8UC1,Scalar(0));
    cv::rectangle(vechile_r, rect, Scalar(255), -1);
    int inter_crowd_region_area = calInterMask(vechile_r,crowd_region);
    if(inter_crowd_region_area*1.0/rect.area()*1.0 > 0.99)
    {
        cv::Mat crowd_region_0 = m_crowd_region_mask[1];
        int inter_crowd_region_area = calInterMask(vechile_r,crowd_region_0);
        if(inter_crowd_region_area > 5)
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


bool CityTracker::is_contain_crowd_region(cv::Rect &detect_box)
{
    cv::Mat crowd_region = m_crowd_region_mask[1];
    cv::Mat vechile_r = cv::Mat (local_param.frame_height,local_param.frame_width,CV_8UC1,Scalar(0));
    cv::rectangle(vechile_r, detect_box, Scalar(255), -1);
    int inter_crowd_region_area = calInterMask(vechile_r,crowd_region);
    if(inter_crowd_region_area > 5)
    {
        return true;
    }
    else
    {
        return false;
    } 
} 



