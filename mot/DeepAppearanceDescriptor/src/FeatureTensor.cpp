
#include "FeatureTensor.h"
#include <iostream>


FeatureTensor *FeatureTensor::instance = NULL;

FeatureTensor *FeatureTensor::getInstance() {
	if(instance == NULL) {
		instance = new FeatureTensor();
	}
	return instance;
}

FeatureTensor::FeatureTensor() {
	//prepare model:
	bool status = init();
	if(status == false)
	  {
	    std::cout<<"init failed"<<std::endl;
	  exit(1);
	  }
	else {
	    std::cout<<"init succeed"<<std::endl;
	  }
}

FeatureTensor::~FeatureTensor() {
	
}

bool FeatureTensor::init() 
{

	feature_dim = 2048;
	return true;
}

bool FeatureTensor::getRectsFeature(DETECTIONS &match_detect_b,
	DETECTIONS &create_detect_box,int frame_id,PARAM_S&params) 
{
	int current_frame_id = frame_id;
	std::string detect_file_path = params.feature_detect_box_file;
	if(current_frame_id < 10)
	{
		detect_file_path = detect_file_path + "img00000" + std::to_string(current_frame_id)  + ".json";
	}
	else if((current_frame_id >=10)&&(current_frame_id < 100))
	{
		detect_file_path = detect_file_path + "img0000" + std::to_string(current_frame_id)  + ".json";
	}
	else if((current_frame_id >=100)&&(current_frame_id < 1000))
	{
		detect_file_path = detect_file_path + "img000" + std::to_string(current_frame_id)  + ".json";
	}
	else if((current_frame_id >=1000)&&(current_frame_id < 10000))
	{
		detect_file_path = detect_file_path + "img00" + std::to_string(current_frame_id)  + ".json";
	}
	else
	{
		std::cerr<<"error frame id"<<std::endl;
	}

	std::vector<cv::Rect> boxes;
	std::vector<float> scores;
	DETECTIONS entir_detect;
	bool ret = parse_file(detect_file_path,entir_detect,params.match_score,boxes,scores,params);
	if(!ret)
	{
		std::cerr<<"Fail to parse file"<<detect_file_path<<std::endl;
		return false;
	}
	if(entir_detect.size() != boxes.size())
	{
		std::cerr<<"error parse_file"<<std::endl;
		return false;
	}
	if(scores.size() != boxes.size())
	{
		std::cerr<<"error parse_file box scores"<<std::endl;
		return false;
	}
	std::vector<int> indexes_l;
    cv::dnn::NMSBoxes(boxes, scores, params.match_score, params.create_score_iou, indexes_l);
    
	std::vector<int> indexes;
    cv::dnn::NMSBoxes(boxes, scores, params.create_score, params.create_score_iou, indexes);
    for (size_t i = 0; i < indexes.size(); i++) 
	{
		DETECTION_ROW dbox;
		int index = indexes[i];
		dbox.tlwh = entir_detect[index].tlwh;
		dbox.is_feature = true;
		dbox.feature = entir_detect[index].feature;
		create_detect_box.push_back(dbox);
    }

	for(size_t j = 0; j < entir_detect.size();j++)
	{
		std::vector<int>::iterator iter = std::find(std::begin(indexes), std::end(indexes), j);
		if(iter == std::end(indexes))
		{
			std::vector<int>::iterator iter_m = std::find(std::begin(indexes_l), std::end(indexes_l), j);
			if(iter_m == std::end(indexes_l))
			{
				DETECTION_ROW dbox;
				dbox.tlwh = entir_detect[j].tlwh;
				dbox.is_feature = false;
				//dbox.feature = entir_detect[j].feature;
				match_detect_b.push_back(dbox);
			}
			else
			{
				DETECTION_ROW dbox;
				dbox.tlwh = entir_detect[j].tlwh;
				dbox.is_feature = true;
				dbox.feature = entir_detect[j].feature;
				match_detect_b.push_back(dbox);
			}
			
		}
	}

	return true;
}
bool FeatureTensor::parse_file(std::string file_path,DETECTIONS &match_detect_b,
float match_score,std::vector<cv::Rect>&rect_list,std::vector<float>&scores,PARAM_S&params)
{
	std::ifstream t(file_path.c_str());
    std::string str((std::istreambuf_iterator<char>(t)),
                  std::istreambuf_iterator<char>());
    rapidjson::Document document;
	if(str.empty())
	{
		std::cerr<<"file is empty:"<<file_path<<std::endl;
		return false;
	}
	document.Parse(str.c_str());
	for(rapidjson::Value::ConstMemberIterator it=document.MemberBegin(); it!= document.MemberEnd();++it)
	{
        std::string key = std::string(it->name.GetString());
		const rapidjson::Value& chileValue = it->value;
		if(chileValue.IsObject())
		{
			DETECTION_ROW dbox;
			rapidjson::Value::ConstMemberIterator chileIter = chileValue.FindMember("conf");
			if(chileIter != chileValue.MemberEnd())
			{
				float conf =  chileIter->value.GetFloat();
				if(conf < match_score)
				{
					continue;
				}
				scores.push_back(conf);
			}
			//std::cout<<"bbox"<<std::endl;
			chileIter = chileValue.FindMember("bbox");
			if(chileIter != chileValue.MemberEnd())
			{
				const rapidjson::Value& box =  chileIter->value;
				if(box.IsArray())
				{
					if(box.Size() == 4)
					{
						int x1 = box[0].GetInt();
						int y1 = box[1].GetInt();
						int x2 = box[2].GetInt();
						int y2 = box[3].GetInt();
						cv::Rect rect = cv::Rect(x1,y1,(x2-x1),(y2-y1));
						if(rect.area() < params.min_rect_eare)
						{
							scores.erase(scores.begin()+(scores.size()-1));
							continue;
						}
						rect_list.push_back(rect);

						DETECTBOX ret = DETECTBOX(rect.x, rect.y,rect.width,  rect.height);
						DETECTION_ROW temp;
						dbox.tlwh = ret;
					}
					else
					{
						std::cerr<<"box error:  "<<file_path<<"id:"<<key<<std::endl;
					}	
				}
				else
				{
					std::cerr<<"box is not array error:  "<<file_path<<"id:"<<key<<std::endl;
				}
				
			}
			chileIter = chileValue.FindMember("feat");
			if(chileIter != chileValue.MemberEnd())
			{
				const rapidjson::Value& feature =  chileIter->value;
				if(feature.IsArray())
				{
					if(feature.Size() != 2048)
					{
						std::cerr<<"feature error:  "<<file_path<<"id:"<<key<<"size:"<<feature.Size()<<std::endl;
						continue;
					}
					std::vector<float> positiveData;
					std::vector<float> normalizedData_l2;
					for(rapidjson::SizeType i = 0; i < feature.Size(); ++i)
					{
						float temp = feature[i].GetFloat();
						positiveData.push_back(temp);
					}
					cv::normalize(positiveData, normalizedData_l2, 1.0, 0, cv::NORM_L2);
					for(size_t i = 0; i < normalizedData_l2.size(); ++i)
					{
						
						// printf("temp:%10lf",temp);
						// exit(0);
						//std::cout<<"feature: "<<temp<<std::endl;
						dbox.feature[i] = normalizedData_l2[i];
					}
					dbox.is_feature = false;
				}
				else
				{
					std::cerr<<"feature is not array error:  "<<file_path<<"id:"<<key<<std::endl;
				}
				
			}
			match_detect_b.push_back(dbox);
		} 
    }
	return true;
}	
void FeatureTensor::tobuffer(const std::vector<cv::Mat> &imgs, uint8 *buf) 
{
	return;
}
void FeatureTensor::test()
{
	return;
}
