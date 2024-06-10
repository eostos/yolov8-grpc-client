/*
 * Eventqueue.hpp
 *
 *  Created on: Nov 26, 2020
 *      Author: ceoebenezer
 */

#ifndef EVENTQUEUE_HPP_
#define EVENTQUEUE_HPP_

#include <string.h>         //strlen
#include <string>           //string
#include <iostream>
#include <sstream>          //stringstream
#include <iomanip>          //setprecision
#include <mutex>
#include <vector>
#include <unistd.h>
#include <cstdlib>

#include <experimental/filesystem>
#include <boost/filesystem.hpp>
#include <boost/filesystem/fstream.hpp>
#include <jsoncpp/json/json.h>

#include <stdio.h>
#include <sys/time.h>
#include <sys/stat.h>

#include "Utils.hpp"

#include <opencv2/opencv.hpp>

using namespace std;
typedef long int lint;

////////////////////////
//// Type: FrameMsg
////////////////////////
struct FrameMsg {
    string  host_uuid = "default-host-uuid";
    string  timestamp = "0";
    string  frame_id = "timestamp-host_uuid";
    float   fps = 0.0;
    int     resolution_x = 0;
    int     resolution_y = 0;
    cv::Mat mat_frame;
    Json::Value analytics_results;
    string analytic_type = "default-type";
    string event_type = "default-type";


    void printEvent() {
        cout << "-------PRINT EVENT-------" << endl;
        cout << "host_uuid:     " << host_uuid << endl;
        cout << "frame_id:      " << frame_id << endl;
        cout << "timestamp:     " << timestamp << endl;
        cout << "fps:           " << fps << endl;
        cout << "resolution_x:  " << resolution_x << endl;
        cout << "resolution_y:  " << resolution_y << endl;
        cout << "mat_frame.sz : " << mat_frame.size() << endl;
        cout << "analytics_results : " << analytics_results << endl;
        cout << "analytic_type : " << analytic_type << endl;
        cout << "event_type : " << event_type << endl;
    }
};

typedef FrameMsg FrMs;
////////////////////////
namespace thmsg {
////////////////////////
////EventQueue
////////////////////////
class EventQueue {

private:
    //THIS CLASS ONLY
    string _MSG_PREFIX = "+[T] ThreadedCapture ";
    //
    bool _debugFlag = 0;
    bool _debugMini = 1;
    deque<FrMs> _th_events_buf;
    const int   _NUMITEMS = 20;
    string      _PORT = "";
    bool _runthread = true;
    //bool _freshFlag = false;
    bool _pauseFlag = true;
    int  _ACTIVE_THREADS = 0;
    long _loop_timr{-1};
    //
    std::mutex m_uport;
    std::mutex m_deque;
    std::mutex m_run;
    //std::mutex m_fresh;
    std::mutex m_pause;
    std::mutex m_count;
    std::mutex m_timer;
    //
    void initLooper();
    bool checkB64Mat(cv::Mat &tempMat);
    bool validateItem(FrameMsg &tempEvent);
    void rest(const int &sleep_msec);
public:
    EventQueue();
    virtual ~EventQueue();
    //
    void lockItemDeque() { m_deque.lock(); }
    void unlockItemDeq() { m_deque.unlock(); }
    void setCaptFlag(bool);
    bool getCaptFlag();
    //void setFreshFlag(bool);
    //bool getFreshFlag();
    void setPauseFlag(bool);
    bool getPauseFlag();
    void setLoopTimr(lint);
    lint getLoopTimr();
    void setVidPath(string);
    string getVidPath();
    void incrCounter();
    void decrCounter();
    void prntCounter();
    //
    auto getNUMITEMS() { return _NUMITEMS; }
    uint getListSize();
    void addNewItem(FrameMsg &in_Item);
    FrMs takeOldestItem();
    deque<FrMs> takeAllMsg();

    void printFailure(string message) {
        printf("=========== WARNING ===========\n[%s]", message.c_str());
    }
};
// namespace
}

#endif /* EVENTQUEUE_HPP_ */
