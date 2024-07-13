#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <iostream>

class CircularBuffer {
public:
    CircularBuffer(size_t size) : size(size), index(0), is_full(false) {
        buffer.resize(size);
    }

    void add(const cv::Mat& photo, const std::string& timestamp) {
        buffer[index] = std::make_pair(photo.clone(), timestamp);
        index = (index + 1) % size;
        if (index == 0) {
            is_full = true;
        }
    }

    std::vector<std::pair<cv::Mat, std::string>> get_all() const {
        if (is_full) {
            std::vector<std::pair<cv::Mat, std::string>> result;
            result.insert(result.end(), buffer.begin() + index, buffer.end());
            result.insert(result.end(), buffer.begin(), buffer.begin() + index);
            return result;
        }
        return std::vector<std::pair<cv::Mat, std::string>>(buffer.begin(), buffer.begin() + index);
    }

private:
    size_t size;
    size_t index;
    bool is_full;
    std::vector<std::pair<cv::Mat, std::string>> buffer;
};
