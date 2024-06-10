#ifndef POLIGON_TRACKER_MANAGER_HPP
#define POLIGON_TRACKER_MANAGER_HPP

#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <thread>



#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "CentroidTracker.hpp"


struct PoligonEvent {
    std::string type;
    int lado;
    cv::Point2f point;
    std::string tracker_id;
    int id_object;
};


class PoligonTrackerManager {
public:

    // bool evaluate(Json::Value &results);
    // bool evaluate(Json::Value &results);
    virtual Json::Value evaluate();
    void evaluateArea();
    void evaluateAreaBbox();

    void setPoints(std::vector<cv::Point2f> vp) { _points = vp; _eval_points.resize(0); }
    void setTrackers(std::vector<TrackingObject*> vto) { this->_trackers = vto;_eval_points.resize(0); }

    std::vector <cv::Point2f> getEvalPoints() { return _eval_points; }

    // void setObjectID(int i) { _object_id = i;}
    // int getObjectID() { return _object_id; }

    void setPoligonID(std::string i) { _poligon_id = i; }
    std::string getPoligonID() { return _poligon_id; }

    cv::Size getSize() { return _size; }
    void setSize(cv::Size s) { _size = s;}

    void setDebug(bool b) { _debug = b; }
    float getDebug() { return _debug; }


protected:
    std::vector<cv::Point2f> _points;
    std::vector<TrackingObject * > _trackers;
    std::vector<cv::Point2f> _eval_points;
    double _poligon_area;
    // int _object_id;
    std::string _poligon_id;
    cv::Size _size;
    cv::Point2f _center_point;
    float _debug = false;
};


class LineTrackerManager : public PoligonTrackerManager {
    virtual Json::Value evaluate();
};



float jsignus(float f);
cv::Point2f interceptionPoint(cv::Point2f p0, cv::Point2f p1, cv::Point2f p2, cv::Point2f p3);
bool checkPoint(cv::Point2f p0, cv::Point2f p1, cv::Point2f);
bool intermediate_point(cv::Point2f p0, cv::Point2f p1, cv::Point2f p2, cv::Point2f p3, cv::Point2f &pc);

#endif
