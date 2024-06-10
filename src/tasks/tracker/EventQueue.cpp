/*
 * EventQueue.cpp
 *
 *  Created on: Nov 26, 2020
 *      Author: ceoebenezer
 */

#include "EventQueue.hpp"

using namespace thmsg;

////////////////////////
////ThreadedListener
////////////////////////
EventQueue::EventQueue()  { cout << _MSG_PREFIX << ": " << __FUNCTION__ << endl; }
////////////////////////////////////////////////
////////////////////////////////////////////////
EventQueue::~EventQueue() { cout << _MSG_PREFIX << ": " << __FUNCTION__ << endl; }
////////////////////////////////////////////////
////////////////////////////////////////////////
void EventQueue::setCaptFlag(bool flag) {
    std::lock_guard<std::mutex> guard(m_run);
    _runthread = flag;
}
bool EventQueue::getCaptFlag() {
    std::lock_guard<std::mutex> guard(m_run);
    return _runthread;
}
void EventQueue::setPauseFlag(bool flag) {
    std::lock_guard<std::mutex> guard(m_pause);
    _pauseFlag = flag;
}
bool EventQueue::getPauseFlag() {
    std::lock_guard<std::mutex> guard(m_pause);
    return _pauseFlag;
}
////////////////////////////////////////////////
////////////////////////////////////////////////
void EventQueue::setLoopTimr(lint t_now) {
    std::lock_guard<std::mutex> guard(m_timer);
    _loop_timr = t_now;
}
lint EventQueue::getLoopTimr() {
    std::lock_guard<std::mutex> guard(m_timer);
    return _loop_timr;
}
////////////////////////////////////////////////
////////////////////////////////////////////////
void EventQueue::setVidPath(string input) {
    std::lock_guard<std::mutex> guard(m_uport);
    _PORT = input;
}
string EventQueue::getVidPath() {
    std::lock_guard<std::mutex> guard(m_uport);
    return _PORT;
}
////////////////////////////////////////////////
////////////////////////////////////////////////
void EventQueue::incrCounter() {
    std::lock_guard<std::mutex> guard(m_count);
    _ACTIVE_THREADS++;
}
void EventQueue::decrCounter() {
    std::lock_guard<std::mutex> guard(m_count);
    _ACTIVE_THREADS--;
}
////////////////////////////////////////////////
////////////////////////////////////////////////
void EventQueue::prntCounter() {
    std::lock_guard<std::mutex> guard(m_count);
    cout << "ACTIVE_THREADS " << _ACTIVE_THREADS << endl;
}
////////////////////////////////////////////////
////////////////////////////////////////////////
uint EventQueue::getListSize() {
    std::lock_guard<std::mutex> guard(m_deque);
    return abs(_th_events_buf.size());
}

void EventQueue::addNewItem(FrameMsg &in_Item) {
    std::lock_guard<std::mutex> guard(m_deque);
    _th_events_buf.push_front(in_Item);
    while ((int)abs(_th_events_buf.size())>_NUMITEMS) {
        cout << "POP MESSAGE " << "back id " << _th_events_buf.back().frame_id << endl;
        _th_events_buf.pop_back();
    }
}

FrMs EventQueue::takeOldestItem()
//SWAPS THE OLDEST FRAME (BACK) WITH A DUMMY FRAME
//DELETES THE DUMMY FRAME NOW IN THE DEQUE
//AND RETURNS THE OLDEST FRAME
{
    std::lock_guard<std::mutex> guard(m_deque);
    FrMs out_item;
    //
    if (_debugFlag) {
    cout << " 1 th_frames_deq size: " << _th_events_buf.size() << endl;
    }
    //SWAPS THE OLDEST FRAME WITH A DUMMY FRAME
    swap(out_item, _th_events_buf.back());
    //REMOVES THE DUMMY FRAME
    _th_events_buf.pop_back();
    //
    if (_debugFlag) {
    cout << " 2 th_frames_deq size: " << _th_events_buf.size() << endl;
    }
    //RETURNS THE OLDEST FRAME
    return out_item;
}

deque<FrMs> EventQueue::takeAllMsg()
//RETRIEVES THE WHOLE BUFFER, LEAVES AN EMPTY BUFFER
{
    std::lock_guard<std::mutex> guard(m_deque);
    deque<FrMs> out_item;
    //
    if (_debugFlag)
        cout << " 1 th_frames_deq size: " << _th_events_buf.size() << endl;
    //SWAP THE BUFFER WITH A DUMMY BUFFER
    swap(out_item, _th_events_buf);
    //
    if (_debugFlag)
        cout << " 2 th_frames_deq size: " << _th_events_buf.size() << endl;
    //RETURNS THE OLDEST FRAME
    return out_item;
}

void EventQueue::initLooper() {
    //INITIALIZE THE THREAD FLAGS & DEQUE VARS
    setCaptFlag(true);
    //setFreshFlag(false);
    setPauseFlag(false);
    _th_events_buf.resize(0);
    m_deque.unlock();
}

bool EventQueue::checkB64Mat(cv::Mat &tempFrame) {
    if (tempFrame.data) {
        return 1;
    } else {
        printFailure("Failure in CAPTURE THREAD while capturing from globalCap object");
        cout << "KILLING THREAD" << endl;
        return 0;
    }
}

bool EventQueue::validateItem(FrameMsg &tempEvent) {
    // RETURN TRUE IF ALL CHECKS ARE PASSED
    return true;
}

void EventQueue::rest(const int &sleep_msec) {
    // PRINT ONCE
    if (_debugFlag)
        cout << " [CaptureThread] GONNA SLEEP: " << sleep_msec * 1000 << " msec" << endl;
    // SLEEP
    usleep(sleep_msec * 1000);
}