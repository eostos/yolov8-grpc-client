/*
 * Utils.hpp
 *
 *  Created on: Dec 17, 2019
 *      Author: dayanm
 */

#ifndef UTILS_HPP_
#define UTILS_HPP_

#include <string>
#include <jsoncpp/json/json.h>
#include <fstream>
#include <sstream>
#include <iostream>
// #include <tbb/tbb.h>
#include <sys/time.h>
#include <sys/socket.h>    //socket
#include <arpa/inet.h> //inet_addr
#include <netdb.h> //hostent

#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <boost/filesystem.hpp>
#include <boost/filesystem/fstream.hpp>

namespace fs=boost::filesystem;

using namespace std;
using namespace cv;

#define _USEGPU_ 0

typedef long int lint;
typedef Point2f Pt2f;

// struct thread_data{
//    std::string evidence_video_path;
//    bool start_clock;
//    tbb::tick_count time_begin_camera;
//    Json::Value config_params;
// };

#if _USEGPU_
#include "yolo_v2_class.hpp"
#else
    struct bbox_t {
        unsigned int x, y, w, h;    // (x,y) - top-left corner, (w, h) - width & height of bounded box
        float prob;                    // confidence - probability that the object was found correctly
        unsigned int obj_id;        // class of object - from range [0, classes-1]
        unsigned int track_id;        // tracking id for video (0 - untracked, 1 - inf - tracked object)
        unsigned int frames_counter;// counter of frames on which the object was detected
    };

#endif

const Scalar EB_RED = Scalar(0,0,255);
const Scalar EB_BLU = Scalar(255,0,0);
const Scalar EB_GRN = Scalar(0,255,0);
const Scalar EB_YEL = Scalar(0,255,255);
const Scalar EB_RED2 = Scalar(0,0,155);
const Scalar EB_BLU2 = Scalar(155,0,0);
const Scalar EB_GRN2 = Scalar(0,155,0);
const Scalar EB_YEL2 = Scalar(0,155,155);

struct dnn_bbox{
    Rect bbox;
    float prob;
    uint obj_id;
    string label;
    string photo_object_cutted;
    string uuid;
    string embeddings;//std::vector<float> embeddings;
	string prob_det;
};

struct mouse_params{
    Mat img;
    cv::Point top_corner;
    cv::Point bottom_corner;
};

struct roi_params{
    vector<vector<Point>> roi_1;
    vector<vector<Point>> roi_2;
    vector<vector<Point>> zone_1;
    vector<vector<Point>> zone_2;
    vector<vector<Point>> zone_3;
    vector<vector<Point>> zone_4;

};

Json::Value readConfig(string config_file, string jsonkey, string id);
std::string slurp(std::ifstream& in);
vector<Rect> filterDnnResults(vector<bbox_t> &_dnn_results);
Rect calcWiderROI(Rect org_ROI, float border_scale, Size image_size);
Rect bbox2Rect(const bbox_t &bbox);
vector<dnn_bbox> convert2Detection(vector<mouse_params> &mp);
void drawMouseParams(vector<mouse_params> &mp, Mat &img);
void drawDets(Mat &debug_img, vector<dnn_bbox> detections);
//B64 ENCODE/DECODE
std::string randomString( size_t length );
static inline bool is_base64(unsigned char c);
std::string base64_encode(unsigned char const* bytes_to_encode, unsigned int in_len);
std::string base64_decode(std::string const& encoded_string);
//TIME
lint getTimeMilis();
string time_stamp (const char *format);
std::string currentDateTime();
std::string currentHour();
//
std::vector<Json::Value> get_list_of_json(Json::Value jsonin);
std::string genId();
std::string unixTimeStampStr();
Point getRectCenter(Rect &rect_i);
float distanceP (Point point1, Point point2);
float vec_mag (const Pt2f& pt);
Rect makeEvalRect(Mat &in_img, Point EVALPOI, int sqr_side);
void drawCorners(Mat &drawable, Rect rect_i, Scalar _color, int TH);
roi_params loadRoiConfig(Json::Value &config,Mat &img_ref);
vector<vector<Point>> readConfigPoints(Json::Value &config,Mat &img_ref);
void udp_send_frames(Mat lpr_socket_img, int sockfd_lpr,struct sockaddr_in lpr_serv);
void create_dir(string input);
void save_evidence(cv::Mat img, string media_path ,string dev_id, string frame_id, string &img_path);
void incr_tic(int &tic_general);
Rect bbox_2_Rect(const bbox_t &bbox);
vector<dnn_bbox> detecton2DnnBox(vector<bbox_t> &bboxes,vector<string> class_names);
vector<dnn_bbox> json2DnnBox(Json::Value input_bbox, int width, int height);
void rectBigger_u(Rect& rect, const Size& matSize, const int& value);
vector<string> loadClassNames(string className);
void drawPoly(Mat &img, vector<vector<Point> > &pts, Scalar color, int THK);
Scalar randomColor();
#endif /* UTILS_HPP_ */
