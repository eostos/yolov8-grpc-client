#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <iostream>

class CircularBuffer {
public:
    CircularBuffer(size_t size) : size(size), index(0), is_full(false) {
        buffer.reserve(size);
    }

    void add(const cv::Mat& photo, const std::string& timestamp) {
        if (photo.empty()) {
            std::cerr << "Warning: Trying to add an empty photo to the buffer." << std::endl;
            return;
        }

        // Debug print to show where the photo is being added
        std::cout << "Adding photo with timestamp " << timestamp << " at index " << index << std::endl;

        if (is_full) {
            // Overwrite the oldest photo
            buffer[index] = std::make_pair(photo.clone(), timestamp);
        } else {
            // Add a new photo
            buffer.push_back(std::make_pair(photo.clone(), timestamp));
        }

        // Update the index to point to the next slot
        index = (index + 1) % size;

        // Mark buffer as full if we've wrapped around
        if (index == 0 && !is_full) {
            is_full = true;
        }
    }

    std::vector<std::pair<cv::Mat, std::string>> get_all() const {
        std::vector<std::pair<cv::Mat, std::string>> result;
        if (is_full) {
            result.insert(result.end(), buffer.begin() + index, buffer.end());
            result.insert(result.end(), buffer.begin(), buffer.begin() + index);
        } else {
            result.insert(result.end(), buffer.begin(), buffer.end());
        }
        return result;
    }

    bool search(const std::string& timestamp) const {
        auto all_entries = get_all();
        for (const auto& entry : all_entries) {
            if (entry.second == timestamp) {
                std::cout << "Timestamp REQUIRED FOUND IN VECTOR " << entry.second << ", Photo size: " << entry.first.size() << std::endl;
                return true;
            } else {
                std::cout << "Timestamp REQUIRED FOUND IN VECTOR " << entry.second << ", Photo size: " << entry.first.size() << std::endl;
                std::cout << "THE REQUEST WASN'T FOUND" << std::endl;
            }
        }
        std::cout << "THE REQUEST WASN'T FOUND IN THE BUFFER" << std::endl;
        return false;
    }

private:
    size_t size;
    size_t index;
    bool is_full;
    std::vector<std::pair<cv::Mat, std::string>> buffer;
};