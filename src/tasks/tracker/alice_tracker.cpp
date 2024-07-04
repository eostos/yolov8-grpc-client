//============================================================================
// Name        : alice_skeleton.cpp
// Author      : Ebenezer Techs -- Daniela C.
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include <stdio.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/core/utility.hpp>

#include <cmath>
#include <fstream>
#include <string>
#include <numeric>
#include <string.h>
#include <vector>
#include <dirent.h>
#include <assert.h>

#include <redox.hpp>
#include <msgpack.hpp>

#include "EventQueue.hpp"
#include "CentroidTracker.hpp"
#include "poligonTrackerManager.hpp"

#include <chrono>
#include <future>
using namespace std;
using namespace cv;
using namespace redox;
using namespace std::chrono;

/////////////////////////////////
/////////////////////////////////
string host_id;
bool read_config = false;
thmsg::EventQueue smartQueueEvents;
thmsg::EventQueue smartQueueFrames;
/////////////////////////////////
/////////////////////////////////
Json::Value dataJson;
Redox rdx;
///////////////////////////////
///////////////////////////////
void send_out_imageb64(Redox &rdx,Mat drawings,string host_id);
void join_and_send_outdata_redox(Redox &rdx,
						FrMs msg_n,
						string media_path,
						Json::Value trackers,
						std::string analityc_type,
						std::string event_type,
						bool save_img,
						Mat img_to_save,
						bool send_b64,
						string to_post,
						bool DEBUG);
std::vector<PoligonTrackerManager*> get_poligons_trackers(Json::Value poligon_json);
void free_poligon_trackers(std::vector<PoligonTrackerManager*> poligon_tracker_manager);
/////////////////////////////////
void send_resize_frame_redis(Mat frame, string host_id, float fps , int v_width, int v_height, string motion_method , string unixTimeStamp,string frameId) {

	cv::Mat flat = frame.reshape(1, frame.total()*frame.channels());
	std::stringstream buffer;
	std::vector<std::string> target;

	std::vector<string> coordinates; 

	msgpack::type::tuple<string, vector<uchar>, std::string, string, float, int, int , vector<string> , string>
				src(host_id, flat, unixTimeStamp, frameId,
					fps,
					v_width,
					v_height,
					coordinates,
					motion_method); //1%

	// serialize the object into the buffer.
	// any classes that implements write(const char*,size_t) can be a buffer.

	msgpack::pack(buffer, src); //2%

	std::string messageToSend(buffer.str());    //0.2%
	//cout  << messageToSend << endl;
	string channel = "motion_streaming_" + host_id  ;
	//string channel_data = "motion_data_" + host_id  ;
   // cout << channel << endl;
	Json::FastWriter fastWriter;
	std::string output = fastWriter.write(dataJson);
	rdx.publish(channel,messageToSend); //3%
	//rdx.publish(channel_data,output); //3%
	buffer.seekg(0);
}


////////////
int main(int argc,const char * argv[]) {
	//READ JSON
	string config_path = "config.json";
	string id = "1";
	string media_path = "/opt/alice-media/tracker";
	
	if (argc != 3) {
		cout << "Usage: " << argv[0] << " <config file (.json)>  <id> "<<endl;
	} else {
		config_path = argv[1];
		id = argv[2];
	}
	
	Json::Value config_params = readConfig(config_path,"config", id);
	string input_url = config_params["url_video"].asString();
	string grpc_port = config_params["grpc_port"].asString();
	string grpc_server = config_params["grpc_server"].asString();
	host_id = config_params["host_id"].asString();
	bool DEBUG = config_params["debug"].asBool();
	bool SEND_B64 = config_params["send_b64"].asBool();
	string media_from_config = config_params["media_path"].asString(); 
	if (media_from_config.size() >0) {
		media_path = media_from_config;
	}
	string to_post = config_params["topost"].asString();
	string redis_ip = config_params["server_ip"].asString();
	int redis_port = config_params["server_port"].asInt();
	cout << redis_port << " REDIS PORT " << endl;
	cout << "Config path: " << config_path << ", id: " << id << ", host_id: " << host_id << endl;

	//REDOX
	//CONNECTIONS
	auto subbed = [](const string& topic) {
        cout << ">>> Subscribed to " << topic << endl;
	};
	auto unsubbed = [](const string& topic) {
        cout << ">>> Unsubscribed from " << topic << endl;
	};
	auto err_callback = [](const string& topic, const int &errnum) {
        cerr << ">>> LAM_ERR_CALLBACK " << topic << " errnum: " << errnum << endl;
        exit(-3);
	};
	auto rdxconn = [](const int& state) {
		cout << "> RDX.CONNECT CALLBACK CONNECTED STATE: " << state << endl;
	};
	auto subconn = [](const int& state) {
		cout << "> SUB.CONNECT CALLBACK CONNECTED STATE: " << state << endl;
	};

	//Redox rdx;
	Subscriber sub;
	if (!rdx.connect(redis_ip, redis_port,rdxconn) || !sub.connect(redis_ip, redis_port,subconn)) {
		cout << "Could not connect to port :  " << redis_port << " - " << host_id << endl;
		cout << "Not connection To Redis - " << host_id << endl;
		exit(-2);
	}
	
	//SET REDIS STATUS VARIABLE
	bool status_analytics = false;
	string status_channel = "tracker_status_analytics_"+host_id;
	string update_channel = "tracker_update_analytics_"+host_id;
	if (!rdx.set(status_channel,"0")) {
		cout << "Failed to set status - " << host_id << endl;  
	}

	// -----------------------------------------------------------------------------------
	//LOAD TRACKER OBJECT
	// -----------------------------------------------------------------------------------
	// TRACKERS CONSTRUCTED AND INITIALIZED
	const size_t MAX_NUM_TRACKERS = 50;//originally was in 25
	std::vector<TrackingObject *> trackers(MAX_NUM_TRACKERS);
	for (size_t i = 0; i < MAX_NUM_TRACKERS; ++i) {
		trackers[i] = new TrackingObject();
		trackers[i]->setMaxRouteSize(250);
		trackers[i]->setMaxDisappeared(50);//originally was in 10
	}

	// ADD TRACKER OBJECTS TO TRACKING MANAGER
	CentroidTracker tracking;
	tracking.setTrackingObjects(trackers);

	// -----------------------------------------------------------------------------------

	// -----------------------------------------------------------------------------------
	// POLYGON  MANAGER
	// -----------------------------------------------------------------------------------
	// VECTOR OF POLYGON AND POLYGON MANAGER ARE CONSTRUCTED AND INITIALIZED
	std::vector<PoligonTrackerManager * > poligon_tracker_manager;
	poligon_tracker_manager = get_poligons_trackers(config_params["poligons"]);
	for (int i = 0, m = (int)poligon_tracker_manager.size(); i < m; ++i) {
		poligon_tracker_manager[i]->setTrackers(trackers);
	}

	// -----------------------------------------------------------------------------------
	//REDIS SUBSCRIPTION CALLBACKS
	auto motion_msg = [](const string& topic, const string& msg) {
		std::string str(msg);
		msgpack::object_handle oh = msgpack::unpack(str.data(), str.size());
		// deserialized object is valid during the msgpack::object_handle instance is alive.
		msgpack::object deserialized = oh.get();
		// msgpack::object supports ostream.
		msgpack::type::tuple<std::string, vector<uchar>, std::string, std::string,
							 float, int, int> dst;
		deserialized.convert(dst);
		////////////////////////////////
		FrMs msgi;
		vector<uchar> vmat  = std::get<1>(dst);
		msgi.frame_id       = std::get<3>(dst); //msgi.timestamp + "-" + msgi.host_uuid;
		msgi.resolution_x   = std::get<5>(dst);
		msgi.resolution_y   = std::get<6>(dst);

		//create Mat from vector
		int rows = msgi.resolution_y;
		int cols = msgi.resolution_x;
		if ((int)vmat.size() == rows*cols*3) // check that the rows and cols match the size of your vector
		{
			Mat img2 = Mat(rows, cols, CV_8UC3); // initialize matrix of uchar of 1-channel where you will store vec data
			//copy vector to mat
			memcpy(img2.data, vmat.data(), vmat.size()*sizeof(uchar)); // change uchar to any type of data values that you want to use instead
			msgi.mat_frame = img2;
		} else {
			cout << "vmat.size() " << vmat.size() << endl;
			cout << "cols " << cols << endl;
			cout << "rows " << rows << endl;
		}
		//cout << "no permitido" << cols << endl;
		smartQueueFrames.addNewItem(msgi);
	};
	auto object_msg = [](const string& topic, const string& msg) {
		std::string str(msg);
		Json::Reader jreader;
		Json::Value json_event;
		jreader.parse(str, json_event,false);
		////////////////////////////////
		FrMs msgi;
		msgi.host_uuid      = json_event["host_id"].asString();
		msgi.timestamp      = json_event["unix_itme_stamp"].asString();
		msgi.frame_id       = json_event["frame_id"].asString();
		msgi.fps            = json_event["fps"].asFloat();
		msgi.resolution_x   = json_event["resolution_x"].asInt();
		msgi.resolution_y   = json_event["resolution_y"].asInt();
		msgi.analytics_results = json_event["analytics_results"];
		msgi.analytic_type = json_event["analytic_type"].asString();
		msgi.event_type = json_event["event_type"].asString();   
		smartQueueEvents.addNewItem(msgi);
	};
	auto config_msg = [](const string& topic, const string& msg) {
		std::string str(msg);
		if (msg==host_id) {
			read_config = true;
		}
	};
	//SUBSCRIPTION
	//string motion_channel = "motion_streaming_" + host_id;
	//sub.subscribe(motion_channel, motion_msg, subbed, unsubbed, err_callback);
	string data_channel = "objects_data_" + host_id;
	sub.subscribe(data_channel, object_msg, subbed, unsubbed, err_callback);
	sub.subscribe("tracker_check_config", config_msg);

	Mat imgShow,imgSave,frame;
	int64 thFPS;
	thFPS = getTickCount();
	float fps_calc = -1;
	int events_counter = 0;
	bool RUN = 1;
	bool OBJECT_SEND = 1;
	bool once_flag = true;
	bool save_img = false;
	//////////////////////////
	cout << "Connecting to Camera :  "<< input_url <<endl;
	VideoCapture cap(input_url);

	if (!cap.isOpened()) {
		cout << "The camera is not reacheable :  " <<endl;
		throw int(42);
	}else{
		cout << "Url opened !!! :  " <<endl;
	}
	string motion_method = "KNN";
	FrMs msgi;


	//////////////////GRPC////////////////
 //auto channel = grpc::CreateChannel(grpc_server+":"+grpc_port, grpc::InsecureChannelCredentials());
 //ImageClient client(channel);
	/////////////////////////////////////
	while (RUN) {
		//////////////////////////////////////////////////
		// LOOP WHILE THE EVENTS BUFFER CONTAINS ELEMENTS
		/////////////////////////////////////////////////
	

		////////////////////////////////
		cap>>frame;
		
		
		

		std::string unixTimeStamp = unixTimeStampStr();
		std::string frameId = unixTimeStamp + "_" + host_id;
		msgi.frame_id       = frameId; //msgi.timestamp + "-" + msgi.host_uuid;
		msgi.resolution_x   = frame.cols;
		msgi.resolution_y   =  frame.rows;
		msgi.timestamp = unixTimeStamp;
		//msgi.mat_frame = frame;
		//create Mat from vector
		int rows = msgi.resolution_y;
		int cols = msgi.resolution_x;
		std::cout << "Init " << std::endl;
		//auto future = std::async(std::launch::async, SendFrameAsync, std::ref(client), frame,fps_calc,host_id,frameId,unixTimeStamp);
        std::string reply = "Server replied: " ;//future.get(); // Wait for the async task to finish and get the reply
        std::cout << "Server replied: " << reply << std::endl;
		namedWindow("imgShow",0);
        imshow("imgShow",frame);
        waitKey(1);
		smartQueueFrames.addNewItem(msgi);
		//cout << "loop !!! :  " <<rows<<" y :  x   " <<cols <<endl;
	

		//send_resize_frame_redis(frame, host_id, fps_calc, cols, rows, motion_method, unixTimeStamp, frameId);




		//////////////////////////////////////////////////
		// UPDATE STATUS TO REDIS SERVER
		/////////////////////////////////////////////////
		if (!status_analytics) {
			if (!rdx.set(status_channel,"1")) {
				cout << "Failed to set status - " << host_id << endl;  
			} else {
				status_analytics = true;
			}
		}
		if (read_config) {
			config_params = readConfig(config_path,"config", id);
			DEBUG = config_params["debug"].asBool();
			media_from_config = config_params["media_path"].asString(); 
			if (media_from_config.size() >0) {
				media_path = media_from_config;
			}
			redis_ip = config_params["server_ip"].asString();
			redis_port = config_params["server_port"].asInt();
			cout << "Configuration has changed - " << host_id << endl;
			read_config=false; 

			// --------------------------------------------------------------------
			// DELETE CURRENT POLYGONS
			free_poligon_trackers(poligon_tracker_manager);
			poligon_tracker_manager.clear(); 
			// CREATE NEW POLYGONS
			poligon_tracker_manager = get_poligons_trackers(config_params["poligons"]);
			// --------------------------------------------------------------------
			for (int i = 0, m = (int)poligon_tracker_manager.size(); i < m; ++i) {
				poligon_tracker_manager[i]->setTrackers(trackers);
			}
		}
		//////////////////////////////////////////////////
		// RESTART LOOP IF THERE'S NO EVENT 
		/////////////////////////////////////////////////        
		auto eventsQueue_size = abs(smartQueueEvents.getListSize());
		auto framesQueue_size = abs(smartQueueFrames.getListSize());
		
		if (eventsQueue_size<1 && !usleep(300)) {
			if (framesQueue_size>2) {
				if(OBJECT_SEND){
					OBJECT_SEND = 0;
				}
				//---10 FRAMES WERE SENT FROM MOTION BUT OBJECT DID NOT RESPOND
				FrMs discard_frame = smartQueueFrames.takeOldestItem();
				cout <<"[i] Object is not sending - " << host_id <<  endl;
				cout <<"[i] Frame discarded - " << host_id <<  endl;
			}
			continue;
		}
		if(!OBJECT_SEND){
			
			for(int i =0;i<framesQueue_size;i++){
				FrMs discard_frame = smartQueueFrames.takeOldestItem();
			}
			for(int i =0;i<eventsQueue_size;i++){
				FrMs discard_event = smartQueueEvents.takeOldestItem();
			}			
			OBJECT_SEND = 1;
			cout <<"[i] Object restored, restarting queues - " << host_id <<  endl;
			continue;
		}		
		////////////////////////////////////////////////////////
		// RESTART LOOP IF THERE'S AN EVENT WITHOUT MOTION FRAME
		/////////////////
		///////////////////////////////////////
		FrMs msg_n = smartQueueEvents.takeOldestItem();
		if (framesQueue_size<1){
			cout <<"[i] Object discarded - " << host_id <<  endl;
			continue;
		}
		FrMs frame_n = smartQueueFrames.takeOldestItem();
		bool FLAG = true;
		if (FLAG) {//frame_n.frame_id == msg_n.frame_id
			//Mat detection_img = frame_n.mat_frame;
			if (frame.empty()) {
   			 		cout << "[!] Warning: detection image is empty. Skipping processing for this frame." << endl;
   				 continue; // Skip further processing for this frame
				}

			vector<dnn_bbox> detections= json2DnnBox(msg_n.analytics_results,msg_n.resolution_x,msg_n.resolution_y);
			
			//TRACKING PROCESS
			int64 ti = cv::getTickCount();
			tracking.setDataImages(frame);
			tracking.UpdateObjects(detections, msg_n.frame_id);
			tracking.evalObjects();
			
			for (auto &trk_i: trackers) {
				trk_i->restartPolygons();
			}

			for (int i = 0, m = (int)poligon_tracker_manager.size(); i < m; ++i) {
				poligon_tracker_manager[i]->setSize(frame.size());
				//poligon_tracker_manager[i]->evaluateAreabbox();
				poligon_tracker_manager[i]->evaluateArea();
				
			}

			Json::Value objects_to_report = tracking.reportObjects(save_img);
			tracking.deactivateObjects();
			
			if (DEBUG) {
				cout << " - Number of detections: " << detections.size() << " - " << host_id << endl;
				cout << " - Number of active trackers: " <<  tracking.getActiveTrackers() << " - " << host_id << endl;
			}

			if (!objects_to_report[0].isNull()) {
				//SEND ANALYTICS RESULTS
				if (DEBUG){
					imgSave = tracking.getDetsImage();
				}else{
					imgSave = frame;
				}
				join_and_send_outdata_redox(rdx,msg_n,media_path,objects_to_report,"", "",save_img,imgSave,SEND_B64,to_post,DEBUG);
				//SET LAST UPDATED TIME
				if (!rdx.set(update_channel,to_string(getTimeMilis()))) {
					cout << "Failed to set update time - " << host_id << endl;
				}
			}
			
			int64 tf = cv::getTickCount();

			// MEASURE FPS
			float timer = (cv::getTickCount() - thFPS)*1000/cv::getTickFrequency();
			events_counter++;
			if (timer > 1000) 
			{
				fps_calc = 1000.0*events_counter/timer;
				cout << "[i] FPS: " << to_string(fps_calc) << " - " << host_id << endl;
				thFPS = getTickCount();
				events_counter = 0;
			}

			if (DEBUG) {
				float data_fin = (tf-ti)*1000/cv::getTickFrequency();
				//cout << " - Time elapsed per frame: " << data_fin << "ms - " << host_id << endl;				
				imgShow = tracking.getDrawImage();

				for (size_t i = 0, n = poligon_tracker_manager.size(); i < n; ++i) {
					if (poligon_tracker_manager[i] == nullptr) { continue; }
					std::vector<cv::Point2f> points = poligon_tracker_manager[i]->getEvalPoints();
					for (size_t j = 0, m = points.size(); j < m; ++j) {
						cv::Point2f p0 = points[j];
						//cv::Point2f p1 = points[(j+1)%m];
						cv::Point2f p1 = points[(j+1)%m];
						int tk = ceil(min(imgShow.cols, imgShow.rows) / 200.0);
						cv::line(imgShow, p0, p1, EB_GRN, 1);
					}
				}
			
			namedWindow("imgShow",0);
            imshow("imgShow",imgShow);
            waitKey(1);
			
			send_out_imageb64(rdx,imgShow,msg_n.host_uuid);
			}
		} else {
			cout <<"[!] Check motion and object sync - "<< host_id << endl;
			cout <<"[i] Motion and object discarded - " << host_id <<  endl;
		}
	}
	sub.disconnect(); rdx.disconnect();

	// ------------------------------------------------------------------------------------
	// DELETE TRACKER POINTERS
	// ------------------------------------------------------------------------------------
	for (size_t i = 0; i < MAX_NUM_TRACKERS; ++i) {
		if (trackers[i] != nullptr) {
			trackers[i]->~TrackingObject();
		}
	}
	// ------------------------------------------------------------------------------------

	// ------------------------------------------------------------------------------------
	// DELETE POLYGON POINTERS
	// ------------------------------------------------------------------------------------
	free_poligon_trackers(poligon_tracker_manager);
	// ------------------------------------------------------------------------------------
	return 0;
}

/////////////////////////////////
/////////////////////////////////
void send_out_imageb64(Redox &rdx,Mat drawings,string host_id) {
	
	std::vector<uchar> buf;
	cv::imencode(".jpg", drawings, buf);
	auto *enc_msg = reinterpret_cast<unsigned char*>(buf.data());
	std::string encoded = base64_encode(enc_msg, buf.size());
	string channel = "face_videofordisplay_"+host_id;
	rdx.publish(channel, encoded);
}

void join_and_send_outdata_redox(Redox &rdx,
						FrMs msg_n,
						string media_path,
						Json::Value trackers,
						std::string analityc_type,
						std::string event_type,
						bool save_img,
						Mat img_to_save,
						bool send_b64,
						string to_post,
						bool DEBUG){

	lint current_time = getTimeMilis();
	std::string str_timestamp = std::to_string(current_time);
	
	//PHOTO
	Json::Value final_json;
	final_json["host_id"] = msg_n.host_uuid;
	final_json["unix_itme_stamp"] = str_timestamp;
	final_json["frame_id"] = msg_n.frame_id;
	final_json["fps"] = msg_n.fps;
	final_json["resolution_x"] = msg_n.resolution_x;
	final_json["resolution_y"] = msg_n.resolution_y;
	final_json["topost"] = to_post;
	final_json["analytics_results"] = trackers;      //may work better with "json_str"
        if(save_img){
			if(send_b64){
				std::vector<uchar> buf;
				cv::imencode(".jpg", img_to_save, buf);
				auto *enc_msg = reinterpret_cast<unsigned char*>(buf.data());
				std::string encoded = base64_encode(enc_msg, buf.size());
				final_json["base_64_frame"] = encoded;
			}else{
				string url_photo_event = "";
				save_evidence(img_to_save, media_path ,msg_n.host_uuid, msg_n.frame_id,url_photo_event);
				final_json["url_photo_event"] = url_photo_event;
			}
        }
	final_json["analityc_type"] = "analityc_type";
	final_json["event_type"] = "event_type";
	
	if (DEBUG){
		cout << final_json << endl;
	}

	Json::StreamWriterBuilder builder;
	std::string string_output_data = Json::writeString(builder, final_json);
	
	std::string temp_channel = "tracker_streaming_" + msg_n.host_uuid;
	rdx.publish(temp_channel, string_output_data);	

}





std::vector<PoligonTrackerManager*> get_poligons_trackers(Json::Value poligon_json) {

	std::vector<PoligonTrackerManager*> poligon_tracker_manager;
	// se leen cada uno de los objectos poligonos
	std::vector<Json::Value> poligon_objects = get_list_of_json(poligon_json);
	// se redimensiona el tracker para almacenar todos los poligonos
	poligon_tracker_manager.resize(poligon_objects.size());
	// 
	int index_pol = 0;
	for (size_t i = 0, n = poligon_objects.size(); i < n; ++i) {
		// obtienen el id del tracker
		bool check_id = poligon_objects[i].isMember("poligon_id");
		if (!check_id) {
			std::cerr << "ERROR: this poligon does not have an \"poligon_id\" key\n";
			continue;
		}
		std::string poligon_id = poligon_objects[i]["poligon_id"].asString();
		// obtiene los puntos del tracker en formato json
		bool check_id_0 = poligon_objects[i].isMember("points");
		if (!check_id_0) {
			std::cerr << "ERROR: wrong input on poligon id: " << poligon_id << "\n";
			std::cerr << "ERROR: this poligon does not have an \"points\" key\n";
			continue;
		}
		std::vector<Json::Value> poligon_points = get_list_of_json(poligon_objects[i]["points"]);
		std::vector<cv::Point2f> points;
		bool temp_flag = false;
		size_t m = poligon_points.size();
		if (m < 3) {
			std::cerr << "ERROR: wrong input on poligon id: " << poligon_id << "\n";
			std::cerr << "ERROR: number of poligon is lower to 3" << std::endl;
			continue;
		}
		for (size_t j = 0; j < m; ++j) {
			// se leen las coordenadas del punto
			bool check_x = poligon_points[j].isMember("x");
			if (!check_x) {
				std::cerr << "ERROR: wrong input on poligon id: " << poligon_id << "\n";
				std::cerr << "ERROR: point " << (j + 1) << " does no have attribute \"x\"" << std::endl;
				temp_flag = true;
				break;
			}
			float x = poligon_points[j]["x"].asFloat();
			bool check_y = poligon_points[j].isMember("y");
			if (!check_y) {
				std::cerr << "ERROR: wrong input on poligon id: " << poligon_id << "\n";
				std::cerr << "ERROR: point " << (j + 1) << " does no have attribute \"y\"" << std::endl;
				temp_flag = true;
				break;
			}
			float y = poligon_points[j]["y"].asFloat();
			points.push_back(cv::Point2f(x,y));
		}
		if (temp_flag) {
			continue;
		}
		// se construye un nuevo objeto
		poligon_tracker_manager[index_pol] = new PoligonTrackerManager();
		// se configura el objeto 
		poligon_tracker_manager[index_pol]->setPoligonID(poligon_id);
		poligon_tracker_manager[index_pol]->setPoints(points);
		index_pol++;
	}

	return poligon_tracker_manager;

}


void free_poligon_trackers(std::vector<PoligonTrackerManager*> poligon_tracker_manager) {
#if 0
	for (size_t i = 0, n = poligon_tracker_manager.size(); i < n; ++i) {
		if (poligon_tracker_manager[i] != nullptr) {
			poligon_tracker_manager[i]->~PoligonTrackerManager();
		}
	}
#else
	for (auto p:poligon_tracker_manager) {
		delete p;
	}
#endif
}