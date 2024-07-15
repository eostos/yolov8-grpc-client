#include "Yolo.hpp"
#include "YoloNas.hpp"
#include "YOLOv7_NMS.hpp"
#include "Triton.hpp"
#include "TorchvisionClassifier.hpp"


//adding tracker libraries
#include <cmath>
#include <fstream>
#include <string>
#include <numeric>
#include <string.h>
#include <vector>
#include <dirent.h>
#include <assert.h>

#include <redox.hpp>
//#include <msgpack.hpp>

#include "EventQueue.hpp"
#include "bufferPhoto.hpp"
#include "CentroidTracker.hpp"
#include "poligonTrackerManager.hpp"
//////////////////////////
using namespace cv;
using namespace redox;


/////////////////////////////////
/////////////////////////////////
string host_id;
CircularBuffer photo_buffer(30);
string media_path = "/opt/alice-media/tracker";
bool read_config = false;
bool save_img_obj= false;
thmsg::EventQueue smartQueueEvents;
thmsg::EventQueue smartQueueFrames;
/////////////////////////////////
/////////////////////////////////
Json::Value dataJson;
Redox rdx;
///////////////////////////////
///////////////////////////////

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
						bool DEBUG,std::string  unixTimeStamp){

	//lint current_time = getTimeMilis();
	//std::string str_timestamp = std::to_string(current_time);
	
	//PHOTO
	Json::Value final_json;
	final_json["host_id"] = msg_n.host_uuid;
	final_json["unix_itme_stamp"] = unixTimeStamp;
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
				if(save_img_obj){
				string url_photo_event = "";
				save_evidence(img_to_save, media_path ,msg_n.host_uuid, msg_n.frame_id,url_photo_event);
				final_json["url_photo_event"] = url_photo_event;
				}
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



void draw_label(cv::Mat& input_image, const std::string& label, float confidence, int left, int top)
{
    const float FONT_SCALE = 0.7;
    const int FONT_FACE = cv::FONT_HERSHEY_DUPLEX; // Change font type to what you think is better for you
    const int THICKNESS = 2;
    cv::Scalar YELLOW = cv::Scalar(0, 255, 255);

    // Display the label and confidence at the top of the bounding box.
    int baseLine;
    std::string display_text = label + ": " + std::to_string(confidence);
    cv::Size label_size = cv::getTextSize(display_text, FONT_FACE, FONT_SCALE, THICKNESS, &baseLine);
    top = cv::max(top, label_size.height);

    // Top left corner.
    cv::Point tlc = cv::Point(left, top);
    // Bottom right corner.
    cv::Point brc = cv::Point(left + label_size.width, top + label_size.height + baseLine);

    // Draw black rectangle.
    cv::rectangle(input_image, tlc, brc, cv::Scalar(255, 0, 255), cv::FILLED);

    // Put the label and confidence on the black rectangle.
    cv::putText(input_image, display_text, cv::Point(left, top + label_size.height), FONT_FACE, FONT_SCALE, YELLOW, THICKNESS);
}

std::unique_ptr<TaskInterface> createClassifierInstance(const std::string& modelType, const int input_width, const int input_height, const int channels)
{
    if (modelType == "torchvision-classifier")
    {
        return std::make_unique<TorchvisionClassifier>(input_width, input_height, channels);
    }
    else
    {
        return nullptr;
    }
}


std::unique_ptr<TaskInterface> createDetectorInstance(const std::string& modelType, const int input_width, const int input_height)
{
    if (modelType == "yolov7nms")
    {
        //return std::make_unique<YOLOv7_NMS>(input_width, input_height);
        std::cout << "Work in progress..." << std::endl;
        return nullptr;
    }
    else if (modelType.find("yolov") != std::string::npos )
    {
          std::cout << "Work in progress... YOLO V choosen" << std::endl;
        return std::make_unique<Yolo>(input_width, input_height);
    }
    else if (modelType.find("yolonas") != std::string::npos )
    {
        return std::make_unique<YoloNas>(input_width, input_height);
    }        
    else
    {
        return nullptr;
    }
}
void send_out_imageb64(Redox &rdx,Mat drawings,string host_id) {
	cv::Mat resized_frame;
    cv::resize(drawings, resized_frame, cv::Size(360*3, 240*3));
	std::vector<uchar> buf;
	 std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, 95};
	cv::imencode(".jpg", resized_frame, buf,params);
	auto *enc_msg = reinterpret_cast<unsigned char*>(buf.data());
	std::string encoded = base64_encode(enc_msg, buf.size());
	std::cout << "host_id" << host_id  << std::endl;
	string channel = "face_videofordisplay_"+host_id;
	rdx.publish(channel, encoded);
}

std::vector<Result> processSource(const cv::Mat& source, 
    const std::unique_ptr<TaskInterface>& task, 
    const std::unique_ptr<Triton>&  tritonClient, 
    const TritonModelInfo& modelInfo)
{
    auto start = std::chrono::steady_clock::now();
    std::vector<uint8_t> input_data = task->preprocess(source, modelInfo.input_format_, modelInfo.type1_, modelInfo.type3_,modelInfo.input_c_, cv::Size(modelInfo.input_w_, modelInfo.input_h_));
	auto end = std::chrono::steady_clock::now();
    auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "preproccess " << diff << " ms" << std::endl;

	auto start1 = std::chrono::steady_clock::now();
	//auto [infer_results, infer_shapes] = tritonClient->inferAsync(input_data);
	auto [infer_results, infer_shapes] = tritonClient->infer(input_data);
	auto end1 = std::chrono::steady_clock::now();
    auto diff1 = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1).count();
    std::cout << "infer Function Total " << diff1 << " ms" << std::endl;

        // Call your processSource function here
 

	auto start2 = std::chrono::steady_clock::now();
    return task->postprocess(cv::Size(source.cols, source.rows), infer_results, infer_shapes);
	auto end2 = std::chrono::steady_clock::now();
	auto diff2 = std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start2).count();
    std::cout << "infer Function Total " << diff2 << " ms" << std::endl;

}


// Define a function to perform inference on an image
void ProcessImage(const std::string& sourceName,
    const std::unique_ptr<TaskInterface>& task, 
    const std::unique_ptr<Triton>&  tritonClient,
    const TritonModelInfo& modelInfo, 
    const std::vector<std::string>& class_names) {

    std::string sourceDir = sourceName.substr(0, sourceName.find_last_of("/\\"));
    
    cv::Mat image = cv::imread(sourceName);

    if (image.empty()) {
        std::cerr << "Could not open or read the image: " << sourceName << std::endl;
        return;
    }

 
    // Call your processSource function here
    std::vector<Result> predictions = processSource(image, task,  tritonClient, modelInfo);

    for (const Result& prediction : predictions) 
    {
        if (std::holds_alternative<Classification>(prediction)) {
            Classification classification = std::get<Classification>(prediction);
            std::cout << class_names[classification.class_id] << ": " << classification.class_confidence << std::endl; 

        } 
        else if (std::holds_alternative<Detection>(prediction)) 
        {
            Detection detection = std::get<Detection>(prediction);
            cv::rectangle(image, detection.bbox, cv::Scalar(255, 0, 0), 2);
            draw_label(image,  class_names[detection.class_id], detection.class_confidence, detection.bbox.x, detection.bbox.y - 1);
        }
    }    

    std::string processedFrameFilename = sourceDir + "/processed_frame.jpg";
    std::cout << "Saving frame to: " << processedFrameFilename << std::endl;
    cv::imwrite(processedFrameFilename, image);
}
template <typename T>
std::string to_string_with_precision(const T a_value, const int n = 6)
{
	std::ostringstream out;
	out.precision(n);
	out << std::fixed << a_value;
	return out.str();
}
// Define a function to perform inference on a video
void ProcessVideo(const std::string& sourceName,
    const std::unique_ptr<TaskInterface>& task, 
    const std::unique_ptr<Triton>&  tritonClient, 
    TritonModelInfo& modelInfo,  
     const std::vector<std::string>& class_names,const std::string& id,Json::Value config_params) {

////////////////////////////////////////////////////



	string input_url = config_params["url_video"].asString();
	string grpc_port = config_params["grpc_port"].asString();
	string grpc_server = config_params["grpc_server"].asString();
	host_id = config_params["host_id"].asString();
	bool DEBUG = config_params["debug"].asBool();
	bool SEND_B64 = config_params["send_b64"].asBool();
	string media_from_config = config_params["media_path"].asString(); 
	if (media_from_config.size() >0) {
		cout << "The media path found in config file is  " << media_from_config << endl;
		media_path = media_from_config;
	}else {
		cout << "The media path was not found in config file , taking by default  /opt/alice-media/tracker  " << media_path << endl;
	}
	string to_post = config_params["topost"].asString();
	string redis_ip = config_params["server_ip"].asString();
	int redis_port = config_params["server_port"].asInt();
	cout << redis_port << " REDIS PORT " << endl;
	cout << "Config path: " << sourceName << ", id: " << id << ", host_id: " << host_id << endl;

//Redox rdx;


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


	Subscriber sub;
	if (!rdx.connect(redis_ip, redis_port,rdxconn) || !sub.connect(redis_ip, redis_port,subconn)) {
		cout << "Could not connect to port :  " << redis_port << " - " << host_id << endl;
		cout << "Not connection To Redis - " << host_id << endl;
		exit(-2);
	}
//////////////







    	//SET REDIS STATUS VARIABLE
	bool status_analytics = false;
	string status_channel = "tracker_status_analytics_"+host_id;
	string update_channel = "tracker_update_analytics_"+host_id;
	if (!rdx.set(status_channel,"0")) {
		cout << "Failed to set status - " << host_id << endl;  
	}
	const size_t MAX_NUM_TRACKERS = 50;//originally was in 25
	std::vector<TrackingObject *> trackers(MAX_NUM_TRACKERS);
	for (size_t i = 0; i < MAX_NUM_TRACKERS; ++i) {
		trackers[i] = new TrackingObject();///
		trackers[i]->setMaxRouteSize(250);
		trackers[i]->setMaxDisappeared(20);//originally was in 10
	}

	// ADD TRACKER OBJECTS TO TRACKING MANAGER
	CentroidTracker tracking;
	tracking.setTrackingObjects(trackers);
	tracking.setPoligonForSpeed(config_params["poligon_speed"]);
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
	auto config_msg = [](const string& topic, const string& msg) {
		std::string str(msg);
		if (msg==host_id) {
			read_config = true;
		}
	};
	//SUBSCRIPTION
		auto notifications_msg = [](const string& topic, const string& msg) {
		std::string str(msg);
		Json::Reader jreader;
		Json::Value json_event;
		jreader.parse(str, json_event,false);
		////////////////////////////////
		string unix_time_stamp    = json_event["unix_itme_stamp"].asString();
		string requets_uuid    = json_event["requets_uuid"].asString();
		auto photos_and_timestamps = photo_buffer.get_all();
		std::cout << "PHOTO WAS REQUIRED :  SEARCHING IN BUFFER " << unix_time_stamp << " \n";
		bool photo_found =false;
		string channel = "notifications_pub_"+host_id;
    for (const auto& [photo, timestamp] : photos_and_timestamps) {
        //std::cout << "Timestamp REQUIRED FOUND IN VECTOR " << timestamp << ", Photo size: " << photo.size() << " bytes\n";
        // Display the photo
		
		if (timestamp==unix_time_stamp) {
			std::string frameId = unix_time_stamp + "_" + host_id;
			string url_photo_event = "";
			save_evidence(photo, media_path ,host_id, frameId,url_photo_event);
			
			
			
			//cv::imshow("Photo", photo);
			//cv::waitKey(100); // Display each photo for 100 milliseconds
			Json::Value final_json;
			final_json["host_id"] = host_id;
			final_json["unix_itme_stamp"] = unix_time_stamp;	
			final_json["photo_evidence"] = url_photo_event;	
			final_json["requets_uuid"] = requets_uuid;		
			Json::StreamWriterBuilder builder;
			std::string string_output_data = Json::writeString(builder, final_json);
			rdx.publish(channel, string_output_data);
			photo_found=true;
			std::cout << "THE REQUEST WAS FOUND :) " << " \n";
				
		 }

    }
		if (!photo_found) {
			
			std::cout << "THE REQUEST WASNT FOUND " << " \n";
			Json::Value final_json;
			Json::StreamWriterBuilder builder;
			std::string string_output_data = Json::writeString(builder, final_json);
			rdx.publish(channel, string_output_data);
		 }
		//// we have to search the vector with time stamp and then record 
		///the foto and then send it into the publisher.

	};
	string notifications_channel = "notifications_sub_"+ host_id;
	sub.subscribe(notifications_channel, notifications_msg, subbed, unsubbed, err_callback);
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
///////////////////////////ALICE///////////////////////////////////////



    std::string sourceDir = media_path;
    std::cout<<" Trying to Open : "<<input_url  <<std::endl;
    cv::VideoCapture cap(input_url);

    if (!cap.isOpened()) {
        std::cout << "Could not open the video: " << input_url << std::endl;
        return;
    }

#ifdef WRITE_FRAME
    cv::VideoWriter outputVideo;
    cv::Size S = cv::Size((int)cap.get(cv::CAP_PROP_FRAME_WIDTH), (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    int codec = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
    outputVideo.open(sourceDir + "/processed.avi", codec, cap.get(cv::CAP_PROP_FPS), S, true);

    if (!outputVideo.isOpened()) {
        std::cout << "Could not open the output video for write: " << input_url << std::endl;
        return;
    }
#endif
	if (!cap.isOpened()) {
		cout << "The camera is not reacheable :  " <<endl;
		throw int(42);
	}else{
		cout << "Url opened !!! :  " <<endl;
	}
	string motion_method = "KNN";
	FrMs msgi;
    double fps=0.0;
    while (true) {
/////////////////////////////
  		auto start = std::chrono::steady_clock::now();
		cap.read(frame) ;
		
      //  resize(frame, frame, Size(1280, 720));
		std::string unixTimeStamp = unixTimeStampStr();
		std::string frameId = unixTimeStamp + "_" + host_id;
		//msgi.mat_frame = frame;
//////////////////////////////////
        // Call your processSource function here
        std::vector<Result> predictions = processSource(frame, task, tritonClient, modelInfo);
		//std::vector<Result> predictions;

        //smartQueueFrames.addNewItem(msgi);
#if defined(SHOW_FRAME) || defined(WRITE_FRAME)
        //double fps = 1000.0 / static_cast<double>(diff);
        std::string fpsText = "FPS: " + std::to_string(fps);
		int point_y = frame.cols/2;
        cv::putText(frame, fpsText, cv::Point(10,point_y ), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
 		if (!status_analytics) {
			if (!rdx.set(status_channel,"1")) {

				cout << "Failed to set status - " << host_id << endl;  
			} else {
				status_analytics = true;
			}
		}
 		if (read_config) {
			config_params = readConfig(sourceName,"config", id);
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
		//	// CREATE NEW POLYGONS
			poligon_tracker_manager = get_poligons_trackers(config_params["poligons"]);
			// --------------------------------------------------------------------
		for (int i = 0, m = (int)poligon_tracker_manager.size(); i < m; ++i) 
		{
				poligon_tracker_manager[i]->setTrackers(trackers);
			}
		}
 		//auto eventsQueue_size = abs(smartQueueEvents.getListSize());
		//auto framesQueue_size = abs(smartQueueFrames.getListSize());
        vector<dnn_bbox> detections;
		Json::Value parts;
		FrMs msgi;
		int idx = 0;
		string obj_id = "1";
        //struct dnn_bbox{
        //    Rect bbox;
        //    float prob;
        //    uint obj_id;
        //    string label;
        //    string photo_object_cutted;
        //    string uuid;
        //    string embeddings;//std::vector<float> embeddings;
	    //    string prob_det;
        //    };
	
        for (auto&& prediction : predictions) 
        {
            if (std::holds_alternative<Detection>(prediction)) 
            {
                 Detection detection = std::get<Detection>(prediction);
                //std::cout << "Detecting cars : "  << detection.bbox<<std::endl;
                //std::cout << "Detecting cars : "  << class_names[detection.class_id]<<std::endl;
                //std::cout << "Detecting cars : "  << detection.class_id<<std::endl;
                //std::cout << "Detecting cars : "  << detection.class_confidence<<std::endl;
                ///////
				Json::Value partInfo;
				//Rect ri = dnn_det_i.bbox;
				float FX = (float)1.0/frame.cols;
				float FY = (float)1.0/frame.rows;
				partInfo["bbox"]["x"] = to_string_with_precision(detection.bbox.x * FX, 4);
				partInfo["bbox"]["y"] = to_string_with_precision(detection.bbox.y * FY, 4);
				partInfo["bbox"]["w"] = to_string_with_precision(detection.bbox.width * FX, 4);
				partInfo["bbox"]["h"] = to_string_with_precision(detection.bbox.height * FY, 4);
				partInfo["prob"]      = to_string_with_precision(  detection.class_confidence, 4);
				partInfo["obj_id"] 	= obj_id;
				partInfo["tag"]	=  class_names[detection.class_id];
				parts[idx] = partInfo;
				idx++;						
				////
                uint id = stoul(obj_id);//this was a fake 
				//std::cout << "Detected class  "  << class_names[detection.class_id]<<std::endl;
				
                dnn_bbox dnn_obj = dnn_bbox{detection.bbox,detection.class_confidence, id, class_names[detection.class_id],"photo_object_cutted","uuid","embeddings",std::to_string(detection.class_confidence)};
                detections.push_back(dnn_obj);
				
                //cv::rectangle(frame, detection.bbox, cv::Scalar(255, 0, 0), 2);
                //draw_label(frame,  class_names[detection.class_id], detection.class_confidence, detection.bbox.x, detection.bbox.y - 1);
            }
        }

		///*Filling the object of detection for send it to web//
				msgi.host_uuid      =host_id;
				msgi.timestamp      =unixTimeStamp;
				msgi.frame_id       =frameId;
				msgi.fps            = fps;
				msgi.resolution_x   = frame.cols;
				msgi.resolution_y   = frame.rows;
				msgi.analytics_results = parts;
				msgi.analytic_type = "analityc_typu";
				msgi.event_type = "event_typu";   
				




			int64 ti = cv::getTickCount();
			auto starttrack = std::chrono::steady_clock::now();
			tracking.setDataImages(frame);
           // UpdateObjects(vector<dnn_bbox> _detections, string frame_id)
			tracking.UpdateObjects(detections,frameId,fps);
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
					//imgSave = tracking.getDetsImage();
					imgShow = tracking.getDrawImage();//this is for debug 
				}else{
					imgSave = frame;
				}
				photo_buffer.add(imgShow, unixTimeStamp);///this is for testing 
			//	photo_buffer.add(imgSave, unixTimeStamp);///this is for testing 
				join_and_send_outdata_redox(rdx,msgi,media_path,objects_to_report,"", "",save_img,imgShow,SEND_B64,to_post,DEBUG,unixTimeStamp);//this is for debug 
				//join_and_send_outdata_redox(rdx,msgi,media_path,objects_to_report,"", "",save_img,imgSave,SEND_B64,to_post,DEBUG);
				//SET LAST UPDATED TIME
				if (!rdx.set(update_channel,to_string(getTimeMilis()))) {
					cout << "Failed to set update time - " << host_id << endl;
				}
			}
								auto endtrack = std::chrono::steady_clock::now();
        	auto difftrack = std::chrono::duration_cast<std::chrono::milliseconds>(endtrack - starttrack).count();
        	std::cout << "Infer time track: " << difftrack << " ms" << std::endl;
			if (DEBUG) {
				//float data_fin = (tf-ti)*1000/cv::getTickFrequency();
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

			//cv::imshow("video feed", imgShow);
        		//cv::waitKey(1);
			send_out_imageb64(rdx,imgShow,msgi.host_uuid);


			}




#endif

#ifdef SHOW_FRAME

		//send_out_imageb64(rdx,frame,msgi.host_uuid); //it takes a lot of time in my pc core I5 around 13 ms 
		auto end2 = std::chrono::steady_clock::now();
        auto diff2 = std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start).count();
        std::cout << "Total Infer time : " << static_cast<double>(diff2) << " ms" << std::endl;
		fps = 1000.0 / static_cast<double>(diff2);

        //cv::imshow("video feed", frame);
        //cv::waitKey(1);
#endif

#ifdef WRITE_FRAME
        outputVideo.write(frame);
#endif
    }
}


static const std::string keys =
    "{ help h   | | Print help message. }"
    "{ model_type mt | yolov7 | yolo version used i.e yolov5, yolov6, yolov7, yolov8}"
    "{ model m | yolov7-tiny_onnx | model name of folder in triton }"
    "{ task_type tt | | detection, classification}"
    "{ source s | data/dog.jpg | path to video or image}"
    "{ serverAddress  ip  | localhost | server address ip, default localhost}"
    "{ port  p  | 8001 | Port number(Grpc 8001, Http 8000)}"
    "{ verbose vb | false | Verbose mode, true or false}"
    "{ protocol pt | grpc | Protocol type, grpc or http}"
    "{ labelsFile l | ../coco.names | path to coco labels names}"
    "{ batch b | 1 | Batch size}"
    "{ input_sizes is | | Input sizes (channels width height)}";

int main(int argc, const char* argv[])
{


///////////////////////////ALICE

    string config_path = "config.json";
	string id = "1";
	
	
	if (argc != 3) {
		cout << "Usage: " << argv[0] << " <config file (.json)>  <id> "<<endl;
	} else {
		config_path = argv[1];
		id = argv[2];
	}
	Json::Value config_params = readConfig(config_path,"config", id);
	
	




/////////////////////////

/*
	string input_url = config_params["url_video"].asString();
	string grpc_port = config_params["grpc_port"].asString();
	string grpc_server = config_params["grpc_server"].asString();
	host_id = config_params["host_id"].asString();
	bool DEBUG = config_params["debug"].asBool();
	bool SEND_B64 = config_params["send_b64"].asBool();
	string media_from_config = config_params["media_path"].asString(); 
	if (media_from_config.size() >0) {
		cout << "The media path found in config file is  " << media_from_config << endl;
		media_path = media_from_config;
	}else {
		cout << "The media path was not found in config file , taking by default  /opt/alice-media/tracker  " << media_path << endl;
	}
	string to_post = config_params["topost"].asString();
	string redis_ip = config_params["server_ip"].asString();
	int redis_port = config_params["server_port"].asInt();
	cout << redis_port << " REDIS PORT " << endl;
	cout << "Config path: " << sourceName << ", id: " << id << ", host_id: " << host_id << endl;
*/






////////////////////////

    cv::CommandLineParser parser(argc, argv, keys);
    if (parser.has("help")) {
        parser.printMessage();
        return 0;
    }

    std::cout << "Current path is " << std::filesystem::current_path() << '\n';
    std::string serverAddress =config_params["grpc_server"].asString();// parser.get<std::string>("serverAddress");
    std::string port = config_params["grpc_port"].asString();//parser.get<std::string>("port");
    bool verbose = config_params["verbose"].asBool();//  parser.get<bool>("verbose");
    std::string sourceName =config_path;  // Changed from 'video' to 'source'
    ProtocolType protocol = ProtocolType::GRPC;//parser.get<std::string>("protocol") == "grpc" ? ProtocolType::GRPC : ProtocolType::HTTP;
    const size_t batch_size = parser.get<size_t>("batch");

    std::string modelName = config_params["model"].asString();//parser.get<std::string>("model");
    std::string modelVersion = "";
    std::string modelType = config_params["model_type"].asString();//parser.get<std::string>("model_type");

    //if(!parser.has("task_type"))
    //{
    //    std::cerr << "Task type (classification or detection) is required " << std::endl;
    //    std::exit(1);
    //}

    std::string taskType = config_params["task_type"].asString();//parser.get<std::string>("task_type");
    std::string url(serverAddress + ":" + port);
    std::string labelsFile = config_params["labelsFile"].asString();//parser.get<std::string>("labelsFile");

    std::string input_sizes_str = config_params["input_sizes"].asString();//parser.get<std::string>("input_sizes");
    std::istringstream input_sizes_stream(input_sizes_str);
    
    size_t input_size_c, input_size_w, input_size_h;
    input_sizes_stream >> input_size_c >> input_size_w >> input_size_h;

    std::cout << "Chosen Parameters:" << std::endl;
    std::cout << "task_type (tt): " << taskType << std::endl;
    std::cout << "model_type (mt): " <<modelType<< std::endl;
    std::cout << "model (m): " << modelName << std::endl;
    std::cout << "source (s): " << sourceName << std::endl;  // Changed from 'video' to 'source'
    std::cout << "serverAddress (ip): " << serverAddress << std::endl;
    std::cout << "verbose (vb): " << verbose << std::endl;
    std::cout << "protocol (pt): " << parser.get<std::string>("protocol") << std::endl;
    std::cout << "labelsFile (l): " << labelsFile << std::endl;
    std::cout << "batch (b): " << parser.get<size_t>("batch") << std::endl;

    std::vector<int64_t> input_sizes;
    if (parser.has("input_sizes")) {
        std::cout << "input_size_c: " << input_size_c << std::endl;
        std::cout << "input_size_w: " << input_size_w << std::endl;
        std::cout << "input_size_h: " << input_size_h << std::endl;
        input_sizes.push_back(batch_size);
        input_sizes.push_back(input_size_c);
        input_sizes.push_back(input_size_w);
        input_sizes.push_back(input_size_h);
    }

   
    

    // Create Triton client
    std::unique_ptr<Triton> tritonClient = std::make_unique<Triton>(url, protocol, modelName);
    tritonClient->createTritonClient();

    TritonModelInfo modelInfo = tritonClient->getModelInfo(modelName, serverAddress, input_sizes);
    std::unique_ptr<TaskInterface> task;
    if (taskType == "detection") {
        task = createDetectorInstance(modelType, modelInfo.input_w_, modelInfo.input_h_);
        if(task == nullptr)
        {
            std::cerr << "Invalid model type specified: " +  modelType << std::endl;
        }
    } 
    else if (taskType == "classification") {
 
        task = createClassifierInstance(modelType, modelInfo.input_w_, modelInfo.input_h_, modelInfo.input_c_);
        if(task == nullptr)
        {
            std::cerr << "Invalid model type specified: " +  modelType << std::endl;
        }
    } 
    else {
        std::cerr << "Invalid task type specified: " +  taskType << std::endl;
        return 1;
    }

    const auto class_names = task->readLabelNames(labelsFile);



    // Get the directory of the source file
    std::string sourceDir = sourceName.substr(0, sourceName.find_last_of("/\\"));
    if (sourceName.find(".jpg") != std::string::npos || sourceName.find(".png") != std::string::npos) {
         ProcessImage(sourceName, task, tritonClient, modelInfo, class_names);
    } else {
         ProcessVideo(sourceName, task, tritonClient, modelInfo, class_names,id,config_params);
    }

    return 0;
}
