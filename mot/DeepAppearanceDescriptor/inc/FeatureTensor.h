
#ifndef _CITY_FEATURE_H
#define _CITY_FEATURE_H
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "rapidjson/document.h"
#include "rapidjson/writer.h"
#include "rapidjson/stringbuffer.h"

#include "model.h"
#include "dataType.h"
#include "city_tracker.h"

typedef unsigned char uint8;

class FeatureTensor
{
public:
	static FeatureTensor* getInstance();
	bool getRectsFeature(DETECTIONS &match_detect_b,
	DETECTIONS &create_detect_box,int frame_id,PARAM_S&params);
private:
	FeatureTensor();
	FeatureTensor(const FeatureTensor&);
	FeatureTensor& operator = (const FeatureTensor&);
	static FeatureTensor* instance;

	bool parse_file(std::string file_path,DETECTIONS &match_detect_b,
float match_score,std::vector<cv::Rect>&rect_list,std::vector<float>&scores,PARAM_S&params);
	bool generate(DETECTIONS&match_detect_b,DETECTIONS&create_detect_box);
	
	~FeatureTensor();

	void tobuffer(const std::vector<cv::Mat> &imgs, uint8 *buf);

	bool init();
	int feature_dim;
	
public:
	void test();
};
#endif