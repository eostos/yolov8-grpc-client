/*
 * Utils.cpp
 *
 *  Created on: Dec 17, 2019
 *      Author: dayanm
 */

#include "Utils.hpp"

static const std::string base64_chars =
     "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
     "abcdefghijklmnopqrstuvwxyz"
     "0123456789+/";

std::string slurp(std::ifstream& in) {
    std::stringstream sstr;
    sstr << in.rdbuf();
    return sstr.str();
}

Json::Value find_obj(Json::Value &obj_src, string key, string desired_value) {
    Json::Value obj_out;
    bool FOUND = false;
    for (auto &obj_i: obj_src) {
        if (obj_i[key] == desired_value) {
            obj_out = obj_i;
            FOUND = true;
            break;
        }
    }
    if (!FOUND)
        cout << endl << "---[ERROR FINDING JSON KEY]---" << endl << endl;
    return obj_out;
}
std::string unixTimeStampStr(){
	lint current_time = getTimeMilis();
	std::string generated_id = std::to_string(current_time);
	return generated_id;
}
Json::Value readConfig(string config_file, string jsonkey, string id) {
    string json_str;
    std::ifstream json_file(config_file, std::ifstream::binary);
    json_str = slurp(json_file);
    Json::Reader jreader;
    Json::Value obj_full, obj_conf,obj_params,obj_id;
    jreader.parse(json_str, obj_full);

    // SEARCH THE LPR CONFIG
    obj_conf = find_obj(obj_full, "type", jsonkey);
    obj_params = obj_conf["params"];

    for (auto &obj_i: obj_params) {
        if (obj_i["id"] == id ) {
            obj_id = obj_i;
        }
    }

    if (obj_id.isNull()){
        cout << endl << "---[ERROR FINDING JSON ID]---" << endl << endl;
    }
    return obj_id;
}

vector<Rect> filterDnnResults(vector<bbox_t> &_dnn_results) {
    vector<uint> ALLOWED_CLASSES {2 , 3 , 5 , 7 };//FOR VOC {5, 6, 13} //YOLO V3 - COCO {2 , 3 , 5 , 7 }; // car,motor bike,bus,truck
    vector<Rect> dnn_rects;
    for (auto &bbox_i: _dnn_results) {
        bool AUTH = 0;
        //LOOK FOR AUTHORIZED CLASSES
        for (auto &CLS_i: ALLOWED_CLASSES) {
            if (bbox_i.obj_id == CLS_i) {
                AUTH = 1;
                break;
            }
        }
        //PUSH ONLY AUTHORIZED DETECTION BOXES
        if (AUTH)
            dnn_rects.push_back(bbox2Rect(bbox_i));
    }
    return dnn_rects;
}

Rect calcWiderROI(Rect org_ROI, float border_scale, Size image_size) {
    Rect wider_ROI;
    //EXPAND THE WIDER ROI
    int inc_rght = org_ROI.width  * border_scale;
    int inc_left = inc_rght;
    int inc_up = org_ROI.height * border_scale;
    int inc_dn = inc_up;
    wider_ROI = Rect(
                org_ROI.x - inc_left,
                org_ROI.y - inc_up,
                org_ROI.width + inc_left + inc_rght,
                org_ROI.height+ inc_up + inc_dn);
    //RECTIFY THE INCREASES
    int wider_roi_X2 = (wider_ROI.x + wider_ROI.width);
    int wider_roi_Y2 = (wider_ROI.y + wider_ROI.height);
    int lim_X1 = 0; //TODO: borders are dangerous
    int lim_Y1 = 0;
    int lim_X2 = image_size.width -1;
    int lim_Y2 = image_size.height-1;
    if (wider_ROI.x < lim_X1) {
        wider_ROI.width  -= abs(lim_X1 - wider_ROI.x);
        wider_ROI.x = lim_X1;
    }
    if (wider_ROI.y < lim_Y1) {
        wider_ROI.height -= abs(lim_Y1 - wider_ROI.y);
        wider_ROI.y = lim_Y1;
    }
    if (wider_roi_X2  > lim_X2)
        wider_ROI.width  -= abs(wider_roi_X2 - lim_X2);
    if (wider_roi_Y2  > lim_Y2)
        wider_ROI.height -= abs(wider_roi_Y2 - lim_Y2);
    //OUTPUT
    return wider_ROI;
}

Rect bbox2Rect(const bbox_t &bbox) {
    return Rect(bbox.x, bbox.y, bbox.w, bbox.h);
}

static void onMouse( int event, int x, int y, int, void* param)
{
    Rect boundingBox;
    mouse_params* mp = (mouse_params*)param;
    Mat & img = mp->img;
    switch ( event )
    {
      case EVENT_LBUTTONDOWN:
        //set origin of the bounding box
        //startSelection = true;
        boundingBox.x = x;
        boundingBox.y = y;
        break;
      case EVENT_LBUTTONUP:
        //sei with and height of the bounding box
        boundingBox.width = std::abs( x - boundingBox.x );
        boundingBox.height = std::abs( y - boundingBox.y );
        //paused = false;
        //selectObject = true;
        break;
      case EVENT_MOUSEMOVE:

        if( true /*startSelection && !selectObject*/ )
        {
          //draw the bounding box
          //Mat currentFrame;
          //image.copyTo( currentFrame );
          //rectangle( img, Point( boundingBox.x, boundingBox.y ), Point( x, y ), Scalar( 255, 0, 0 ), 2, 1 );
          circle(img, Point(x,y), 2, Scalar(255,255,255),20,-1);
        }
        break;
    }

}
Rect createDetection(Mat image){
    Rect detection;
    mouse_params mp;
    mp.img = image;
    //namedWindow("Tracking API",0);
    setMouseCallback( "car_det", onMouse, (void *)&mp);
    //imshow( "Tracking API", mp.img );
    return detection;
}

vector<dnn_bbox> convert2Detection(vector<mouse_params> &mp) {
    vector<dnn_bbox> detections;
    for(uint i = 0; i < (mp.size()-1); i++){
        dnn_bbox detection;
        detection.bbox = Rect(mp[i].top_corner,mp[i].bottom_corner);
        detection.prob = 0.99;
        detection.obj_id = 1;
        detection.label = "car";
        detections.push_back(detection);
    }
    mp.resize(0);
    return detections;
}

Rect bbox_2_Rect(const bbox_t &bbox) {
    return Rect(bbox.x, bbox.y, bbox.w, bbox.h);
}

void rectBigger_u(Rect& rect, const Size& matSize, const int& value)
{
    // move & scale on x axis
    rect.x = (rect.x - value) < 0 ? 0 :(rect.x - value);
    int maxWidth = matSize.width - (rect.x + 1);
    int newWidth = rect.width + 2*value;
    rect.width = newWidth > maxWidth ? maxWidth : newWidth;

    // move & scale on y axis
    rect.y = (rect.y - value) < 0 ? 0 :(rect.y - value);
    int maxHeight = matSize.height - (rect.y + 1);
    int newHeight = rect.height + 2*value;
    rect.height = newHeight > maxHeight ? maxHeight : newHeight;

}

vector<dnn_bbox> detecton2DnnBox(vector<bbox_t> &bboxes, vector<string> class_names) {
    vector<uint> ALLOWED_CLASSES {0, 2 , 3 , 5 , 7 };//FOR VOC {5, 6, 13} //YOLO V3 - COCO {2 , 3 , 5 , 7 }; // car,motor bike,bus,truck

    vector<dnn_bbox> detections;

    for (uint i = 0; i < (bboxes.size()); i++) {

            bool AUTH = false;
            //LOOK FOR AUTHORIZED CLASSES
            for (auto &CLS_i: ALLOWED_CLASSES) {
                if (int(bboxes[i].obj_id) == CLS_i) {
                    AUTH = 1;
                    break;
                }
            }

        if (AUTH) {
            dnn_bbox detection;
            detection.bbox = bbox2Rect(bboxes[i]);
            detection.prob = bboxes[i].prob;
            detection.obj_id = bboxes[i].obj_id;
            detection.label = string(class_names[bboxes[i].obj_id]);;
            detections.push_back(detection);
        }
    }
    return detections;
}

vector<dnn_bbox> json2DnnBox(Json::Value input_bbox, int width, int height){
    vector<dnn_bbox> dnnBboxes;
    for (auto &obj_i: input_bbox) {
        
        Json::Value json_bbox = obj_i["bbox"];
        int x = int(stof(json_bbox["x"].asString())*width);
        int y = int(stof(json_bbox["y"].asString())*height);
        int w = int(stof(json_bbox["w"].asString())*width);
        int h = int(stof(json_bbox["h"].asString())*height);
        float prob = stof(obj_i["prob"].asString());
        uint id = stoul(obj_i["obj_id"].asString());
        dnn_bbox dnn_obj = dnn_bbox{Rect(x,y,w,h),prob, id, obj_i["tag"].asString(),obj_i["photo_object_cutted"].asString(),obj_i["uuid"].asString(),obj_i["embeddings"].asString(),obj_i["prob_det"].asString()};
        dnnBboxes.push_back(dnn_obj);
    }
    return dnnBboxes;
}

vector<string> loadClassNames(string className){

    vector<string> classNamesVec;
    ifstream classNamesFile(className);
    if (classNamesFile.is_open())
    {
        string className = "";
        while (std::getline(classNamesFile, className))
            classNamesVec.push_back(className);
    }
    return classNamesVec;
}
void drawMouseParams(vector<mouse_params> &mp, Mat &img) {
    for(uint i = 0; i < mp.size() - 1; i++){
        rectangle( img, mp[i].top_corner, mp[i].bottom_corner, Scalar( 255, 0, 0 ), 2, 1 );
    }
}

void drawDets(Mat &debug_img, vector<dnn_bbox> detections){
    for (auto &det: detections){
        Rect det_bbox = det.bbox;
        rectangle(debug_img,det_bbox,Scalar(255,255,0), 2, 8, 0);   
        
    }
}

std::string randomString( size_t length )
{
    auto randchar = []() -> char
    {
        const char charset[] =
        "0123456789"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz";
        const size_t max_index = (sizeof(charset) - 1);
        return charset[ rand() % max_index ];
    };
    std::string str(length,0);
    std::generate_n( str.begin(), length, randchar );
    return str;
}
static inline bool is_base64(unsigned char c) {
    return (isalnum(c) || (c == '+') || (c == '/'));
}

std::string base64_encode(unsigned char const* bytes_to_encode, unsigned int in_len) {
    std::string ret;
    int i = 0;
    int j = 0;
    unsigned char char_array_3[3];
    unsigned char char_array_4[4];

    while (in_len--) {
        char_array_3[i++] = *(bytes_to_encode++);
        if (i == 3) {
            char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
            char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
            char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
            char_array_4[3] = char_array_3[2] & 0x3f;

            for(i = 0; (i <4) ; i++)
                ret += base64_chars[char_array_4[i]];

            i = 0;
        }
    }

    if (i)
    {
        for(j = i; j < 3; j++)
            char_array_3[j] = '\0';

        char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
        char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
        char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
        char_array_4[3] = char_array_3[2] & 0x3f;

        for (j = 0; (j < i + 1); j++)
            ret += base64_chars[char_array_4[j]];

        while((i++ < 3))
            ret += '=';
    }
    return ret;
}

std::string base64_decode(std::string const& encoded_string) {
    int in_len = encoded_string.size();
    int i = 0;
    int j = 0;
    int in_ = 0;
    unsigned char char_array_4[4], char_array_3[3];
    std::string ret;

    while (in_len-- && ( encoded_string[in_] != '=') && is_base64(encoded_string[in_])) {
        char_array_4[i++] = encoded_string[in_]; in_++;
        if (i ==4) {
            for (i = 0; i <4; i++)
                char_array_4[i] = base64_chars.find(char_array_4[i]);

            char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
            char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
            char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

            for (i = 0; (i < 3); i++)
                ret += char_array_3[i];

            i = 0;
        }
    }

    if (i) {
    for (j = i; j <4; j++)
        char_array_4[j] = 0;

    for (j = 0; j <4; j++)
        char_array_4[j] = base64_chars.find(char_array_4[j]);

    char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
    char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
    char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

    for (j = 0; (j < i - 1); j++) ret += char_array_3[j];
    }

    return ret;
}

void incr_tic(int &tic_general) {
    tic_general = tic_general >= 1024 ? 0 : tic_general+1;
}

lint getTimeMilis() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return tp.tv_sec * 1000 + tp.tv_usec / 1000;
}

string time_stamp (const char *format)
{
    time_t rawtime;
    struct tm * timeinfo;
    char buffer[80];

    time (&rawtime);
    timeinfo = localtime(&rawtime);
    strftime(buffer,80, format,timeinfo);
    string str(buffer);
    return str;
}

std::string currentDateTime() {
   time_t     now = time(0);
   struct tm  tstruct;
   char       buf[80];
   tstruct = *localtime(&now);
   // Visit http://en.cppreference.com/w/cpp/chrono/c/strftime
   // for more information about date/time form at
   strftime(buf, sizeof(buf), "%m_%d_%Y", &tstruct);
   return buf;
}

std::string currentHour() {
   time_t     now = time(0);
   struct tm  tstruct;
   char buf[80];
   tstruct = *localtime(&now);
   // Visit http://en.cppreference.com/w/cpp/chrono/c/strftime
   // for more information about date/time form at
   strftime(buf, sizeof(buf), "%H", &tstruct);
   return buf;
}


std::string genId(){
    lint current_time = getTimeMilis();
    std::string rand_str = randomString(5);
    std::string generated_id = std::to_string(current_time) +"_"+ rand_str;
    return generated_id;
}

Point getRectCenter(Rect &rect_i) {
    return Point((rect_i.x + rect_i.width/2) , (rect_i.y + rect_i.height/2));
}

float distanceP (Point point1, Point point2)
{
    float diffY = point1.y - point2.y;
    float diffX = point1.x - point2.x;
    float dist;
    dist = sqrt((diffY * diffY) + (diffX * diffX));
    return dist;
}

float vec_mag (const Pt2f& pt) {
    return sqrt( pt.x*pt.x + pt.y*pt.y );
}

Rect makeEvalRect(Mat &in_img, Point EVALPOI, int sqr_side) {
    //PROPOSE A NEW (X1,Y1) AND (X2,Y2) POINTS
    int X1 = EVALPOI.x - sqr_side/2;
    int Y1 = EVALPOI.y - sqr_side/2;
    int X2 = EVALPOI.x + sqr_side/2;
    int Y2 = EVALPOI.y + sqr_side/2;
    //DEFINE THE EVALUATION RECT LIMITS
    int lim_X1 = 0;
    int lim_Y1 = 0;
    int lim_X2 = in_img.cols - 1;
    int lim_Y2 = in_img.rows - 1;
    //RECTIFY THE PROPOSED POINTS
    if (X1 < lim_X1)
        X1 = lim_X1;
    if (Y1 < lim_Y1)
        Y1 = lim_Y1;
    if (X2 > lim_X2)
        X2 = lim_X2;
    if (Y2 > lim_Y2)
        Y2 = lim_Y2;
    //MAKE A REFERENCE TO THE EVALUATION ZONE
    Rect img_eval_rect (Point(X1, Y1) , Point(X2, Y2));
    return img_eval_rect;
}

void drawCorners(Mat &drawable, Rect rect_i, Scalar _color, int TH) {
    uint rectWid = rect_i.width;
    uint rectHei = rect_i.height;
    uint len = min(rectWid, rectHei)/10;
    Point vr1(rect_i.x, rect_i.y);
    Point vr2(vr1.x + rect_i.width, rect_i.y);
    Point vr3(rect_i.x, rect_i.y + rect_i.height);
    Point vr4(vr2.x, vr3.y);
    Point dlx(len, 0);
    Point dly(0, len);
    line(drawable, vr1, vr1+dlx, _color, TH);
    line(drawable, vr1, vr1+dly, _color, TH);
    line(drawable, vr2, vr2-dlx, _color, TH);
    line(drawable, vr2, vr2+dly, _color, TH);
    line(drawable, vr3, vr3+dlx, _color, TH);
    line(drawable, vr3, vr3-dly, _color, TH);
    line(drawable, vr4, vr4-dlx, _color, TH);
    line(drawable, vr4, vr4-dly, _color, TH);

    if (TH==2) {
    int V = (TH+1)/2;
    vr1 += Point(-V,-V);
    vr2 += Point( V,-V);
    vr3 += Point(-V, V);
    vr4 += Point( V, V);
    dlx.x += 2*V;
    dly.y += 2*V;
    line(drawable, vr1, vr1+dlx, _color, 1);
    line(drawable, vr1, vr1+dly, _color, 1);
    line(drawable, vr2, vr2-dlx, _color, 1);
    line(drawable, vr2, vr2+dly, _color, 1);
    line(drawable, vr3, vr3+dlx, _color, 1);
    line(drawable, vr3, vr3-dly, _color, 1);
    line(drawable, vr4, vr4-dlx, _color, 1);
    line(drawable, vr4, vr4-dly, _color, 1);
    }
}

void udp_send_frames(Mat lpr_socket_img, int sockfd_lpr,struct sockaddr_in lpr_serv) {
    //  vector<int> p-;
    //  p.push_back(cv::IMWRITE_JPEG_QUALITY);
    //  p.push_back(95);
        //time_t rawtime_;
        Mat res_original;
        resize(lpr_socket_img,res_original, Size(320,180));
        vector<uchar> lpr_dataFrame;
        socklen_t lpr_m = sizeof(lpr_serv);

        imencode(".jpg", res_original, lpr_dataFrame);
        string lpr_encodedData = base64_encode(&lpr_dataFrame[0], lpr_dataFrame.size()); //lpr data to send
    //  time (&rawtime_);
    //  struct tm * timeinfo;
    //  timeinfo = localtime (&rawtime_); // datetime

        //int result_udp_lpr = sendto(sockfd_lpr,lpr_encodedData.c_str(),strlen(lpr_encodedData.c_str()) ,0,(struct sockaddr *)&lpr_serv,lpr_m);
        sendto(sockfd_lpr,lpr_encodedData.c_str(),strlen(lpr_encodedData.c_str()) ,0,(struct sockaddr *)&lpr_serv,lpr_m);
}

void create_dir(string input) {
    fs::path p(input);
    if(!fs::exists(p)){
        try {
            fs::create_directories(p);
        } catch (fs::filesystem_error &e) {
            cerr << e.what() << endl;
        }
    }
}

void save_evidence(cv::Mat img, string media_path ,string dev_id, string frame_id, string &img_path) {

    fs::path media_p(media_path);
    if(fs::exists(media_p)){
        string folder_day_path =  media_path + "/" + currentDateTime();
        create_dir(folder_day_path);
        string folder_tracker_path =  folder_day_path + "/"+dev_id;
        create_dir(folder_tracker_path);
        string folder_hour_path = folder_tracker_path + "/" + currentHour();
        create_dir(folder_hour_path);
        img_path = folder_hour_path + "/" + frame_id + "_fullImg.jpg";   
        cv::imwrite(img_path, img);
    }else{
        cout << "[!] MEDIA PATH DOES NOT EXIST, EVIDENCE IS NOT SAVED" << endl;
    }
}


vector<vector<Point>> readConfigPoints(Json::Value &config,Mat &img_ref){
    vector<Point> roi_1;
    vector<Point> zone_1;
    vector<Point> zone_2;
    vector<Point> roi_2;
    vector<Point> zone_3;
    vector<Point> zone_4;

    vector<vector<Point>> grouped_points;

    //@Read Roi coordinates and fit in image reference
    for(auto &point: config["roi1"]["roi_coord"]){
        if(!point.empty()){
            Point coord;
            coord.x = int(point[0].asFloat()*img_ref.cols);
            coord.y = int(point[1].asFloat()*img_ref.rows);
            roi_1.push_back(coord);
        }
    }

    for(auto &point: config["roi1"]["zone1"]){
        if(!point.empty()){
            Point coord;
            coord.x = int(point[0].asFloat()*img_ref.cols);
            coord.y = int(point[1].asFloat()*img_ref.rows);
            zone_1.push_back(coord);
        }
    }

    for(auto &point: config["roi1"]["zone2"]){
        if(!point.empty()){
            Point coord;
            coord.x = int(point[0].asFloat()*img_ref.cols);
            coord.y = int(point[1].asFloat()*img_ref.rows);
            zone_2.push_back(coord);
        }
    }

    for(auto &point: config["roi2"]["roi_coord"]){
        if(!point.empty()){
            Point coord;
            coord.x = int(point[0].asFloat()*img_ref.cols);
            coord.y = int(point[1].asFloat()*img_ref.rows);
            roi_2.push_back(coord);
        }
    }

    for(auto &point: config["roi2"]["zone3"]){
        if(!point.empty()){
            Point coord;
            coord.x = int(point[0].asFloat()*img_ref.cols);
            coord.y = int(point[1].asFloat()*img_ref.rows);
            zone_3.push_back(coord);
        }
    }

    for(auto &point: config["roi2"]["zone4"]){
        if(!point.empty()){
            Point coord;
            coord.x = int(point[0].asFloat()*img_ref.cols);
            coord.y = int(point[1].asFloat()*img_ref.rows);
            zone_4.push_back(coord);
        }
    }

    grouped_points.push_back(roi_1);
    grouped_points.push_back(zone_1);
    grouped_points.push_back(zone_2);
    grouped_points.push_back(roi_2);
    grouped_points.push_back(zone_3);
    grouped_points.push_back(zone_4);

    return grouped_points;
}


roi_params loadRoiConfig(Json::Value &config,Mat &img_ref) {
    roi_params roi_config;

    vector<vector<Point>> config_points;
    vector<vector<vector<Point>>> config_contours;

    config_points = readConfigPoints(config,img_ref);

    for(auto &area : config_points){
        vector<vector<Point>> contour;
        Mat src = Mat::zeros( img_ref.size(), CV_8U );
        for(unsigned int i = 0; i< area.size();i++){
            line( src, area[i],  area[(i+1)%area.size()], Scalar( 255 ), 3 );
            /*namedWindow("src",0);
            imshow("src",src);
            waitKey(0);*/
        }
        if(area.size()>0)
            findContours( src, contour, RETR_TREE, CHAIN_APPROX_SIMPLE);
        config_contours.push_back(contour);
    }
    //vector<vector<Point>> vec2(config_contours[0]);
    //copy(vec2.begin(), vec2.end(), back_inserter(roi_config.roi_1));
    //roi_config.roi_1.resize(0);
    //roi_config.roi_1.assign(config_contours[0].begin(),config_contours[0].end());
    roi_config.roi_1    = config_contours[0];
    roi_config.zone_1   = config_contours[1];
    roi_config.zone_2   = config_contours[2];
    roi_config.roi_2    = config_contours[3];
    roi_config.zone_3   = config_contours[4];
    roi_config.zone_4   = config_contours[5];

//  Mat src = Mat::zeros( img_ref.size(), CV_8U );
//  cv::fillPoly(src,roi_config.roi_1,Scalar(255));
//  namedWindow("src",0);
//  imshow("src",src);
//  waitKey(0);
    return roi_config;
}

void drawPoly(Mat &img, vector<vector<Point> > &pts, Scalar color, int THK) {
    for(auto &area : pts){
        for(unsigned int i = 0; i< area.size();i++){
            line( img, area[i],  area[(i+1)%area.size()], color, THK);
            /*namedWindow("src",0);
            imshow("src",src);
            waitKey(0);*/
        }
        break;
    }
}

Scalar randomColor(){
    return Scalar(255,rand()%256,rand()%256);
}
