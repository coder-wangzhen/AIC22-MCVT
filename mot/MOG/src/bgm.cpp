#include "bgm.h"


SegmBg::SegmBg(int init_frame_num,std::string image_path):m_image_path(image_path),m_init_frame_num(init_frame_num)
{
    useRoi = false;
}

SegmBg::~SegmBg()
{

}

GpuMat SegmBg::createMat(Size size, int type,bool useRoi)
{
    Size size0 = size;

    GpuMat d_m(size0, type);

    return d_m;
}

GpuMat SegmBg::loadMat(const Mat& m, bool useRoi)
{
    GpuMat d_m = createMat(m.size(), m.type(), useRoi);
    d_m.upload(m);
    return d_m;
}

RET_INFO SegmBg::init_model(std::string image_path,int m_init_frame_numl)
{
    RET_INFO ret;
    try
    {

        mog2 = cv::cuda::createBackgroundSubtractorMOG2();
        mog2->setDetectShadows(true);
        
        std::vector<cv::String> image_files;
        cv::glob(image_path, image_files);
        if(image_files.size() < m_init_frame_numl)
        {
            std::cout<<"warning: The number of images created model is less "<<std::endl;
            m_init_frame_numl = image_files.size();
        }
        if(image_files.size() > 0)
        {
            cpu_foreground = cv::imread(image_files[0]);
        }
        else
        {
            std::cout<<"ERROR:Fail to init model,images files is empty"<<std::endl;
            ret.message = "ERROR:Fail to init model,images files is empty";
            ret.ret = false;
            return ret;
        }
        gpu_fore_ground = createMat(cpu_foreground.size(), CV_8UC1, useRoi);

        cv::Mat frame;

        for(size_t i = 0; i < m_init_frame_numl;i++)
        {
           frame = cv::imread(image_files[i]);
    
           mog2->apply(loadMat(frame, useRoi), gpu_fore_ground);
        }

        gpu_back_ground = createMat(cpu_foreground.size(), cpu_foreground.type(), useRoi);
        mog2->getBackgroundImage(gpu_back_ground);
        return ret;

    }
    catch (cv::Exception& e)
	{
        std::cerr << e.what() << '\n';
        ret.message = e.what();
        ret.ret = false;
        return ret;
	}
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        ret.message = e.what();
        return ret;
    }
}

RET_INFO SegmBg::get_foreground_mask(cv::Mat &frame,cv::Mat &mask,float learing_rate)
{
    RET_INFO ret;
    try
    {
        mog2->apply(loadMat(frame, useRoi), gpu_fore_ground,learing_rate);
        gpu_fore_ground.download(mask);
        return ret;
        
    }
    catch (cv::Exception& e)
	{
        std::cerr << e.what() << '\n';
        ret.message = e.what();
        ret.ret = false;
        return ret;
	}
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        ret.message = e.what();
        return ret;
    }
    
}



RET_INFO SegmBg::get_background_image(cv::Mat &back_ground_image)
{
    RET_INFO ret;
    try
    {
        gpu_back_ground = createMat(cpu_foreground.size(), cpu_foreground.type(), useRoi);
        mog2->getBackgroundImage(gpu_back_ground);
        gpu_back_ground.download(back_ground_image);
        return ret;
    }   
    catch (cv::Exception& e)
	{
        std::cerr << e.what() << '\n';
        ret.message = e.what();
        ret.ret = false;
        return ret;
	}
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        ret.message = e.what();
        return ret;
    } 
    
}

