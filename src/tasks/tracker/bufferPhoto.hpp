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
        if (photo.empty()) {
            std::cerr << "Warning: Trying to add an empty photo to the buffer." << std::endl;
            return;
        }
        buffer[index] = std::make_pair(photo.clone(), timestamp);
        index = (index + 1) % size;
        if (index == 0) {
            is_full = true;
        }
    }

    std::vector<std::pair<cv::Mat, std::string>> get_all() const {
        std::vector<std::pair<cv::Mat, std::string>> result;
        if (is_full) {
            result.insert(result.end(), buffer.begin() + index, buffer.end());
            result.insert(result.end(), buffer.begin(), buffer.begin() + index);
        } else {
            result.insert(result.end(), buffer.begin(), buffer.begin() + index);
        }
        return result;
    }

private:
    size_t size;
    size_t index;
    bool is_full;
    std::vector<std::pair<cv::Mat, std::string>> buffer;
};