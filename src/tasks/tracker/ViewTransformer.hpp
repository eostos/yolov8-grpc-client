#include <vector>
#include <opencv2/core.hpp>
#include <numeric>


using namespace std;
using namespace cv;

class ViewTransformer {
public:
    ViewTransformer(const std::vector<cv::Point2f>& source, const std::vector<cv::Point2f>& target) {
        cv::Mat srcMat(source), tgtMat(target);
        transformMatrix = cv::getPerspectiveTransform(srcMat, tgtMat);
    }

    std::vector<cv::Point2f> transform_points(const std::vector<cv::Point2f>& points) {
        std::vector<cv::Point2f> transformed_points;
        cv::perspectiveTransform(points, transformed_points, transformMatrix);
        return transformed_points;
    }

private:
    cv::Mat transformMatrix;
};