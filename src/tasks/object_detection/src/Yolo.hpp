#pragma once
#include "common.hpp"
#include "TaskInterface.hpp"

class Yolo : public TaskInterface {
public:
    Yolo(int input_width, int input_height);

    cv::Rect get_rect(const cv::Size& imgSz, const std::vector<float>& bbox);
    cv::Rect get_rect_2(const cv::Size& imgSz, const std::vector<float>& bbox);
    
    auto getBestClassInfo(std::vector<float>::iterator it, const size_t& numClasses);
    
    std::vector<Result> postprocess(const cv::Size& frame_size, std::vector<std::vector<float>>& infer_results,
                                    const std::vector<std::vector<int64_t>>& infer_shapes) override;

    std::vector<uint8_t> preprocess(const cv::Mat& img, const std::string& format, int img_type1, int img_type3,
                                    size_t img_channels, const cv::Size& img_size) override;

private:
    int input_width_;
    int input_height_;
};
