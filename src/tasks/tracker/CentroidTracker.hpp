#ifndef __CENTROID_TRACKER_H__
#define __CENTROID_TRACKER_H__

#include <vector>
#include <opencv2/core.hpp>

#include "Utils.hpp"

using namespace std;
using namespace cv;

static int MET_NONE = 0;
static int MET_HALF = 1;
static int MET_FLOW = 2;

class TrackingObject {
protected:
    bool active; //alive
    bool awake; // with detection
    bool matched;
    int disappeared; //number of lost frames
    bool to_delete; // tracker to delete
    vector<Point> route;
    vector<dnn_bbox> route_bboxes;
    vector<float> route_areas;
    vector<vector<Point2f>> route_polygons;
    
    int max_route_size;
    int max_disappeared;
    int max_route_draw = 14;//7 ORIGINALLY 
    int SHIFT_METHOD = MET_NONE;

    Rect limit_tracker;
    Rect _bbox;
    Mat img;
    Mat drawable;

    int status;

    int numSent;
    string id;
    uint obj_id;
    string obj_label;
    lint timestamp_in;
    lint timestamp_out;

    float max_distance;
    lint last_updated;
    lint duration;
    Scalar trk_color=(0,0,0);
    string init_frame_id;
    
public:
    TrackingObject();
    virtual ~TrackingObject();

    // jelambrar methods
    virtual bool predict(const cv::Mat & mat, cv::Rect &rect) { return false; }
    vector<string> polygon_inside;
    Json::Value polygon_events;

    // rectangle is very usefull 
    cv::Rect getBBOX() { return _bbox; }
    void setBBOX(cv::Rect r) { _bbox = r; }
    //
    std::vector<Point> getRoute() { return route; }
    std::vector<dnn_bbox> getBboxRoute() { return route_bboxes; }
    std::vector<float> getBboxArea() { return route_areas; }
    std::vector<vector<Point2f>> getBboxPoligon() { return route_polygons; }
    //
    void setMaxRouteSize(int maxRouteSize);
    void setLimitTracker(const Rect &limitTracker);
    void setMaxDisappeared(int maxDisappeared);
    bool isActive() const;
    void setActive(bool active);
    bool isAwake() const;
    void setAwake(bool awake);
    bool isMatched() const;
    void setMatched(bool matched);
    void restartPolygons();
    void setPolygon(string pol_id);
    void setPolygonEvent(Json::Value event_poligon);
    string getId();
    uint getObjId();
    string getTimestampIn();
    string getTimestampOut();
    string getDuration();
    int getNumSent();
    void incNumSent();
    void setStatus(int _status);
    int getStatus();
    void setInitFrameId(string _frame_id);
    string getInitFrameId();
    float getMaxDistance();
    void updateDisappeared();
    int getDisappeared();
    int getRouteSize();
    bool isUpdated();
    bool isToDelete();
    void setToDelete(bool toDelete);
    void setShiftMethod(int METHOD);
    //
    void setTrackingData(Mat &img_trkr, Mat &img_draw);
    float calcDistance(cv::Point center_det);
    float calcDistAlt(cv::Point center_det, int SHIFT_METHOD);
    Point calcProjCenter(int SHIFT_METHOD);
    dnn_bbox getRouteBboxTail();
    //
    vector<Point> getRouteToSend();
    void startTracker(uint _obj_id, string _obj_label);
    void setTrackerDNN(const dnn_bbox &dnnData, cv::Point center_det);
    bool lifeControl();
    void shutDown();
    //
    void drawRoute(Mat &draw_trks);
    void drawSearchRadius(Mat &draw_trks, int SHIFT_METHOD);
    void drawTrack(Mat &draw_trks, int &ix);

};

class CentroidTracker {

protected:
    bool DEBUG = 0;
    int MAX_TRACKERS   = 25;       // Max number of trackers allowed
    int MAX_DIST    = 100;          // Max distance between detections of same object
    int ROUTE_MAX_SIZE = 250;       // Max route size for tracker
    int MAX_DISAPPEARED = 20;      //MAX NUMBER OF CONSECUTIVE FRAMES A GIVEN OBJECT HAS TO BE LOST/DISAPPEARED FOR UNTIL WE REMOVE IT

    Mat draw_img;
    Mat dets_img;
    Mat frame;
    // vector<TrackingObject> objects;
    std::vector<TrackingObject*> objects;
    bool setted_roi = false;
    Rect limit_roi_objects;

    vector<dnn_bbox> detections;
    vector<string> ids_with_dets;

    void drawRoi();
    void setDrawRoi();
    vector<dnn_bbox> validateDets(vector<dnn_bbox> &_detections);

public:
    CentroidTracker();
    virtual ~CentroidTracker();

    void setTrackingObjects(std::vector<TrackingObject *> o) { objects = o; }
    void drawObjects(cv::Mat &mat);

    void setDataImages(Mat &frame);
    virtual void UpdateObjects(vector<dnn_bbox> _detections,string frame_id);
    Mat getDrawImage();
    Mat getDetsImage();
    void evalObjects();
    Json::Value reportObjects(bool &save_img);
    void deactivateObjects();
    int getActiveTrackers();
};
#endif
