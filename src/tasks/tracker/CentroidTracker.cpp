#include "CentroidTracker.hpp"

template <typename T>
std::string to_string_with_precision(const T a_value, const int n = 6)
{
	std::ostringstream out;
	out.precision(n);
	out << std::fixed << a_value;
	return out.str();
}
////////////////////////////////////////////////////////////////////////////////////////
TrackingObject::TrackingObject() {
	active = false;
	awake = false;
	matched = false;
	speed_km_vec = "0";
	speed = 0.0 ;




}

TrackingObject::~TrackingObject() {

}

void TrackingObject::setMaxRouteSize(int maxRouteSize) {
	max_route_size = maxRouteSize;
}

void TrackingObject::setLimitTracker(const Rect &limitTracker) {
	limit_tracker = limitTracker;
}

void TrackingObject::setMaxDisappeared(int maxDisappeared) {
	max_disappeared = maxDisappeared;
}

bool TrackingObject::isActive() const {
	return active;
}

void TrackingObject::setActive(bool active) {
	this->active = active;
}

bool TrackingObject::isAwake() const {
	return awake;
}

void TrackingObject::setAwake(bool awake) {
	this->awake = awake;
}

bool TrackingObject::isMatched() const {
	return matched;
}

void TrackingObject::setMatched(bool matched) {
	this->matched = matched;
}
void TrackingObject::restartPolygons() {
	this->polygon_inside.resize(0);
	this->polygon_events = Json::Value(Json::arrayValue);
}
void TrackingObject::setPolygon(string pol_id) {
	this->polygon_inside.push_back(pol_id);
}
void TrackingObject::setPolygonEvent(Json::Value event_poligon) {
	uint idx = this->polygon_events.size();
	this->polygon_events[idx] = event_poligon;
	//cout<<polygon_events.size()<<endl;
	//for(int i=0;i<polygon_events.size();i++){
	//	cout<<polygon_events[i]<<endl;
	//}
	
}

string TrackingObject::getId() {
	return id;
}

uint TrackingObject::getObjId() {
	return obj_id;
}

string TrackingObject::getTimestampIn() {
	return to_string(timestamp_in);
}

string TrackingObject::getTimestampOut() {
	return to_string(timestamp_out);
}

string TrackingObject::getDuration() {
	return to_string_with_precision(float(duration)/1000.0, 4);
}

int TrackingObject::getNumSent() {
	return numSent;
}

void TrackingObject::incNumSent() {
	numSent++;
}

void TrackingObject::setStatus(int _status) {
	status = _status;
}

int TrackingObject::getStatus() {
	return status;
}

void TrackingObject::setInitFrameId(string _frame_id) {
	init_frame_id = _frame_id;
}

string TrackingObject::getInitFrameId() {
	return init_frame_id;
}

float TrackingObject::getMaxDistance() {
	return max_distance;
}

void TrackingObject::updateDisappeared() {
	disappeared += 1;
	awake = false;
}

int TrackingObject::getDisappeared() {
	return disappeared;
}

int TrackingObject::getRouteSize() {
	return route.size();
}

bool TrackingObject::isUpdated() {
	return (getTimeMilis() - last_updated) < 2000;
}

bool TrackingObject::isToDelete() {
	return to_delete;
}

void TrackingObject::setToDelete(bool toDelete) {
	to_delete = toDelete;
	timestamp_out = getTimeMilis();
	duration = timestamp_out - timestamp_in;
	status = 2;
}

void TrackingObject::setShiftMethod(int METHOD) {
	SHIFT_METHOD = METHOD;
}
////////////////////////////////////////////////////
void TrackingObject::setTrackingData(Mat &img_trkr, Mat &img_draw) {
	img = img_trkr;
	drawable = img_draw;
}
Mat TrackingObject::getDrawImageFromTracker() {
	return drawable;
}
float TrackingObject::calcDistance(cv::Point center_det) {
	float distance = -1;
	if (route.size() > 0) {
		distance = distanceP(route.back(), center_det);
	}
	return distance;
}

float TrackingObject::calcDistAlt(cv::Point center_det, int SHIFT_METHOD) {
	float distance = -1;
	if (route.size() > 0) {
		Point proj_center = calcProjCenter(SHIFT_METHOD);
		distance = distanceP(proj_center, center_det);
	}
	return distance;
}

Point TrackingObject::calcProjCenter(int SHIFT_METHOD) {
	Point delta(0, 0);
	if (SHIFT_METHOD == MET_NONE) {
		delta = Point(0, 0);
	} else
	if (SHIFT_METHOD == MET_HALF) {
		if (route.size() >= 2) {
			Point pt_prev = route[route.size()-2];
			Point pt_last = route[route.size()-1];
			delta = (pt_last - pt_prev)/2;
			//Point proj_center = pt_last + delta;
		}
	}
	// WHAT IF IT FALLS OUTSIDE OF THE FRAME???
	return delta + route.back();
}

dnn_bbox TrackingObject::getRouteBboxTail() {
	return route_bboxes.back();
}

vector<Point> TrackingObject::getRouteToSend() {
	vector<Point> route_to_send;
	for (int j=1; j<route.size(); j++) {
		if (j > route.size()-max_route_draw) {
			route_to_send.push_back(route[j]);
		}
	}
	return route_to_send;
}
////////////////////////////////////////////////////////////////////////////////////////
void TrackingObject::startTracker(uint _obj_id, string _obj_label) {
	obj_id = _obj_id;
	obj_label = _obj_label;
	active  = true;
	awake = false;
	matched = false;

	disappeared = 0;
	to_delete = false;
	
	status = 0;

	route.resize(0);
	route_bboxes.resize(0);
	route_areas.resize(0);
	route_polygons.resize(0);
	numSent = 0;

	id  = genId();
	timestamp_in = getTimeMilis();  

	trk_color = EB_BLU;//randomColor();<<

	setShiftMethod(MET_HALF);
}




void TrackingObject::updateSpeed(double fps, const std::vector<cv::Point2f> speed_polygon, const std::vector<cv::Point2f> speed_meters, ViewTransformer view_transformer) {
    // Draw the speed polygon for visualization
    for (const auto& point : speed_polygon) {
        cv::circle(drawable, point, 5, cv::Scalar(0, 0, 255), -1); // Draw red circles
    }
    for (size_t i = 0; i < speed_polygon.size(); ++i) {
        cv::line(drawable, speed_polygon[i], speed_polygon[(i + 1) % speed_polygon.size()], cv::Scalar(255, 0, 0), 2); // Draw blue lines
    }

    // Check if the current position is within the polygon
    double result = cv::pointPolygonTest(speed_polygon, current_position, false);

    if (result > 0) {
        
        std::vector<cv::Point2f> points_to_transform = { current_position };
        std::vector<cv::Point2f> transformed_points = view_transformer.transform_points(points_to_transform);

        if (!transformed_points.empty()) {
            cv::Point2f transformed_point = transformed_points[0];

            // Apply a simple moving average for smoothing
            if (!speed_coordinates.empty() && std::abs(speed_coordinates.back().y - transformed_point.y) > 5) {
                // Ignore large jumps
                transformed_point.y = speed_coordinates.back().y;
            }
            speed_coordinates.push_back(transformed_point);

            if (speed_coordinates.size() > fps/2) {
                speed_coordinates.pop_front(); // Maintain the deque size

                cv::Point2f coordinate_start = speed_coordinates.back();
                cv::Point2f coordinate_end = speed_coordinates.front();
                float distance = cv::norm(coordinate_start - coordinate_end); // Euclidean distance

                //std::cout << "Distance: (" << distance << ")" << std::endl;

                // Draw previous and current positions for visualization
                cv::circle(drawable, previous_position, 5, cv::Scalar(0, 0, 255), -1); // Red for previous position
                cv::circle(drawable, current_position, 5, cv::Scalar(0, 255, 0), -1); // Green for current position

                double time = static_cast<double>(speed_coordinates.size()) / fps;
                double speed_mps = distance / time; // Speed in meters per second
                speed_kmh = static_cast<int>(speed_mps * 3.6); // Convert to km/h
                speed_km_vec = std::to_string(speed_kmh);
                std::string speed_text = std::to_string(speed_kmh) + " km/h";
                //std::cout << "Speed: " << speed_kmh << " km/h, Distance: " << distance << ", Coordinates Size: " << speed_coordinates.size() << " Transformed Point: " << transformed_point << " ID: " << getId() << " Current Position: " << current_position << std::endl;
				std::cout <<  "Speed: " << speed_kmh << std::endl;
                cv::putText(drawable, speed_text, cv::Point(current_position.x, current_position.y - 10), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 0, 255), 3);


            }
        }
    }
}


//(185.30645161290323, 438.4677419354839), (1124.6612903225805, 438.4677419354839), (861.4354838709676, 312.01612903225805), (355.67786187322616, 311.4772942289499)


Point TrackingObject::getCurrentPosition() const {
    return current_position;
}
void TrackingObject::setTrackerDNN(const dnn_bbox &dnnData, cv::Point center_det) {
	///////

	////
	
    previous_position = current_position;
    // Update current position
    current_position = center_det;

	//std::cout << "speed_is_DNN  " <<center_det<<"   " <<getId()<<"   " <<current_position <<std::endl;
	

//}else{
/*
	ViewTransformer view_transformer(source, target);
	std::vector<cv::Point2f> points_to_transform = { static_cast<cv::Point2f>(center_det) };
	std::vector<cv::Point2f> transformed_points = view_transformer.transform_points(points_to_transform);

    // Extract the first point from transformed_points and convert to cv::Point
    cv::Point transformed_point(static_cast<int>(transformed_points[0].x), static_cast<int>(transformed_points[0].y));
    // Calculate the overall distance in meters
	////
    previous_position = current_position;
    // Update current position
    current_position = transformed_point;
}
*/


	///
	
	route.push_back(center_det);
	route_bboxes.push_back(dnnData);
	
	Rect bbox = dnnData.bbox;
	route_areas.push_back(bbox.area());

	route_polygons.push_back({Point(bbox.x,bbox.y),Point(bbox.x+bbox.width,bbox.y),Point(bbox.x+bbox.width,bbox.y+bbox.height),Point(bbox.x,bbox.y+bbox.height)});

	max_distance = bbox.width<bbox.height ? bbox.width/2 : bbox.height/2;
	//
	int min_side = min(bbox.width, bbox.height);
	int max_side = max(bbox.width, bbox.height);
	if (max_side > 2*min_side) {
		max_distance += (max_side/2 - min_side)/2;
		//min(bbox.width, bbox.height)/2 + abs(bbox.width - box.height)/2
	}
	//
	//If gun -> max_distance*4
	if(obj_label=="bag"){max_distance *= 3;}

	last_updated = getTimeMilis();
	duration = last_updated - timestamp_in;

	if (!awake) {
		awake = true;       
	}
}

bool TrackingObject::lifeControl() {
	bool C1 = !limit_tracker.contains(route.back()); //Point is inside the frame big roi
	bool C2 = disappeared > max_disappeared; //Missed more than X frames
	bool C3 = route.size() > max_route_size; //Long enough
	bool C4 = !(isUpdated()); //Lost tracker for N secs
	//bool C4 = (getTimeMilis() - last_updated) > 2000; //Lost tracker for N secs

	if (C1 || C2 || C3 || C4) {
		return true; //Object is out of bounds or lost, cannot be tracked.
	} else {
		return false; //Object is inside of bounds, can be tracked still.
	}
}

void TrackingObject::shutDown() {
	active          = false;
	awake           = false;
	matched 		= false;
	disappeared     = 0;
	to_delete       = false;
	route.resize(0);
	route_bboxes.resize(0);
	route_areas.resize(0);
	route_polygons.resize(0);
	///
speed_buffer.resize(0);
speed_coordinates.resize(0);
speed=0;
speed_km_vec = "";
	///
	numSent 		= 0;
	id 				= "";
	timestamp_in 	= 0;
	timestamp_out 	= 0;
	duration 		= 0;
}
////////////////////////////////////////////////////////////////////////////////////////
void TrackingObject::drawRoute(Mat &draw_trks) {
	//Point ini = route.size() > max_route_draw ? route[route.size()-max_route_draw] : route.front();
	Point ini = route.back();
	Point nex;

	Scalar LIN_COLOR = isMatched() ? Scalar(EB_RED) : EB_RED/2;
	int ix = 0;
	int ROUTE_SIZ = route.size();
	uint limit = max(0,ROUTE_SIZ-max_route_draw);
	for (int j=ROUTE_SIZ-1; j>limit; j--) {
		LIN_COLOR[2] = max(LIN_COLOR[2]-(ix++)*10 , 0.0);
		nex = route[j-1];
		line (draw_trks, ini, nex, LIN_COLOR, 2);
		ini = nex;
	}
}

void TrackingObject::drawSearchRadius(Mat &draw_trks, int SHIFT_METHOD) {

	float searchRadius = getMaxDistance();
	Scalar qtyColor = isAwake() ? EB_GRN : EB_RED;
	Point proj_center = route.back();

	if (SHIFT_METHOD == MET_NONE) {
		proj_center = route.back();
	} else {
		proj_center = calcProjCenter(SHIFT_METHOD);
		line(draw_trks, route.back(), proj_center, Scalar(255,255,50), 2);
	}

	circle(draw_trks, proj_center, searchRadius, qtyColor, 2);
}

void TrackingObject::drawTrack(Mat &draw_trks, int &ix) {
	// TAIL POINT
	Point tail_pt = route.back();
	dnn_bbox tail_rect = route_bboxes.back();
	Rect last_box = tail_rect.bbox;
	Scalar BOX_COLOR = isMatched() ? trk_color : Scalar(110,110,110);
	Point pt0(last_box.x, last_box.y-10);
	Point pt1(last_box.x, last_box.y+30);
	string strprob = tail_rect.label + " " + to_string_with_precision(tail_rect.prob, 2);
	string strid   = this->getId();
	Scalar TXT_COLOR = isMatched() ? EB_GRN : EB_RED;
	rectangle(draw_trks, last_box, BOX_COLOR, 2, 8, 0);
	drawRoute(draw_trks);
	//putText(draw_trks, strid, pt0, FONT_HERSHEY_SIMPLEX, 0.75, TXT_COLOR, 2); //it daw the tracker 
	//putText(draw_trks, strprob, pt0, FONT_HERSHEY_SIMPLEX, 0.75, TXT_COLOR, 2);
}

// ---------------------------------------------------------------------
// ---------------------------------------------------------------------
CentroidTracker::CentroidTracker() 
    : view_transformer(std::vector<cv::Point2f>(4), std::vector<cv::Point2f>(4)) { // Initialize with default 4-point vectors
    // Default constructor - other initializations if necessary
	detect_speed = false;
}

CentroidTracker::~CentroidTracker() {
    // Destructor body - clean up resources if necessary
}

void CentroidTracker::setTransformedImage() {
	
    view_transformer = ViewTransformer(speed_poligon, speed_meters);
	detect_speed = true;
}

void CentroidTracker::drawRoi() {
	if (!setted_roi) {
		float x1,x2,y1,y2;
		x1 = 0.05;x2=0.95;y1=0.05;y2=0.95;
		limit_roi_objects = Rect (Point(draw_img.cols * x1, draw_img.rows * y1),
						Point(draw_img.cols * x2, draw_img.rows * y2) );
		setted_roi = true;
		for (auto &obj: objects) {
			obj->setLimitTracker(limit_roi_objects);
		}
	}
	setDrawRoi();
}

void CentroidTracker::setDrawRoi() {
	Rect trkrDetROI = limit_roi_objects;
	trkrDetROI.x+=1;
	trkrDetROI.y+=1;
	trkrDetROI.width-=2;
	trkrDetROI.height-=2;
	int WID = draw_img.cols;
	int HEI = draw_img.rows;
	draw_img(Rect(0, 0, WID, trkrDetROI.y)) *= 0.5;
	draw_img(Rect(Point(0, trkrDetROI.y + trkrDetROI.height) , Point(WID, HEI))) *= 0.5;
	draw_img(Rect(0, trkrDetROI.y, trkrDetROI.x, trkrDetROI.height)) *= 0.5;
	draw_img(Rect(Point(trkrDetROI.x + trkrDetROI.width, trkrDetROI.y) ,
				   Point(WID, trkrDetROI.y + trkrDetROI.height))) *= 0.5;
}

void CentroidTracker::setDataImages(Mat &frame) {
	this->frame = frame;
	this->draw_img = frame.clone();
	this->dets_img = frame.clone();
	drawRoi();
	for (auto &obj: objects) {
		obj->setTrackingData(this->frame,this->draw_img);
	}
}

vector<dnn_bbox> CentroidTracker::validateDets(vector<dnn_bbox> &_detections) {
	vector<dnn_bbox> valid_dets;
	for (auto &det: _detections) {
		bool valid = true;
		string msg = "VALIDATE ";
		Rect det_bbox = det.bbox;

		if (det_bbox.width==0) {
			msg.append(det.label + "---> Width=0 ");
			det_bbox.width = 20;
		}
		if (det_bbox.height==0) {
			det_bbox.height = 20;
			msg.append("Height = 0");
		}

		Point rect_center = getRectCenter(det.bbox);
		if (!limit_roi_objects.contains(rect_center)){
			valid = false;
		}
		if (valid) {
			valid_dets.push_back(det);
		}
	}
	return valid_dets;
}
double CentroidTracker::getFps(){

return totalTime;

}
void CentroidTracker::UpdateObjects(vector<dnn_bbox> _detections, string frame_id, double fps ,bool DEBUG) {
    /*
    @ Draw Search Radius
    */
    //for (auto &trk_i: objects) {
    //    if (trk_i->isActive()) {
            //trk_i->drawSearchRadius(draw_img);
    //    }
    //}
    /*
    @ Add all Detections to active trackers
    */
   
    detections = validateDets(_detections);
    for(auto detection: detections){
        Rect det_bbox = detection.bbox;
       // cv::rectangle(dets_img, det_bbox, Scalar(255,64,64), 4, 8, 0);
    }
    // Set all trackers as NOT matched
    for (auto &trk_i: objects) {
        if (trk_i->isActive()) {
            trk_i->setMatched(false);
        }
    }
    /*
    @ Update disappeared time if there are not detections
    */  
    if (detections.size() < 1) {
        for (auto &trk_i: objects) {
            if (trk_i->isActive()) {
                trk_i->updateDisappeared();
            }
        }
    } else {
        //SEARCH DETS NEAR TO EACH OBJECT
        //DEFINE THE INERTIA METHOD TO SHIFT THE CENTER OF THE SEARCH AREA
        int METHOD = MET_HALF;
        for (auto &trk_i: objects) {
            if (trk_i->isActive()) {
                float min_dist = 10000;
                int nearest_idx = -1;
                bool is_found = false;
                //trk_i->drawSearchRadius(draw_img, METHOD);
                //- PROYECTAR UN CENTRO VIRTUAL USANDO LA TRAYECTORIA PREVIA DE ROUTE (try flow)
                //- AQUÍ SERÍA POSIBLE USAR MINIFLOW
                for (int i=0; i<detections.size();i++) {
                    Point center_det = getRectCenter(detections[i].bbox);
                    //- vvvv ÉSTE CALCULO SE DEBE HACER CON EL CENTRO PROYECTADO
                    //float dist = trk_i->calcDistance(projected_center);
                    float dist = trk_i->calcDistAlt(center_det, METHOD);
                    //float dist = distanceP(projected_center, center_det);
                    float max_dist = trk_i->getMaxDistance();
                    bool C1 = (dist<min_dist);
                    bool C2 = (dist<max_dist);
                    bool C3 = (detections[i].obj_id == trk_i->getObjId());
    /*
    //treat truck as car
    string aux_id0 = detections[i].label;
    string aux_id1 = trk_i->getRouteBboxTail().label;
    //aux0 = aux0==7 ? 2 : aux0;
    //aux1 = aux1==7 ? 2 : aux1;
    //bool C3 = (aux_id0 == aux_id1);
    if ( ( aux_id0=="car" && aux_id1=="truck" ) || ( aux_id0=="truck" && aux_id1=="car" ) ) {
        C3 = 1;
    cout << " xx aux_id0 " << aux_id0 << endl;
    cout << " xx aux_id1 " << aux_id1 << endl;
    }*/
                    bool C4 = (trk_i->isUpdated());
                    bool C5 = !(trk_i->isMatched());
                    if (C1 && C2 && C3 && C5) {
                        min_dist = dist;
                        is_found = true;
                        nearest_idx = i;
                    }
                }
                // IF FOUND, REMOVE THE DETECTION_i, AND UPDATE THE TRACKER
                if (is_found) {
                    trk_i->setTrackerDNN(detections[nearest_idx], getRectCenter(detections[nearest_idx].bbox));
                    trk_i->setStatus(1);
                    trk_i->setMatched(true);
                    // Calculate speed in pixels/frame
					if(detect_speed){
                    	trk_i->updateSpeed(fps,speed_poligon,speed_meters,view_transformer);
                    }
					detections.erase(detections.begin() + nearest_idx);
                //OTHERWISE, INCREMENT HE DISAPP COUNTER
                } else {
                    trk_i->updateDisappeared();
                }
            }
        }
        // UNMATCHED DETECTIONS WILL TRIGGER THE CREATION OF NEW TRACKERS
        for (auto &detection: detections) {
            if (detection.prob > 0.1) {
                for (auto &trk_i: objects) {
                    if (!trk_i->isActive()) {
                        trk_i->startTracker(detection.obj_id,detection.label);
                        trk_i->setInitFrameId(frame_id);
                        trk_i->setTrackerDNN(detection, getRectCenter(detection.bbox));
                        cout << "*-*-*-*-*-*-* Create tracker : "<< trk_i->getId() <<endl;
                        break;
                    }
                }
            }
        }
    }

    /*
    @ Draw Images
    */
    
    // Dibuja trayectorias y estado solo si DEBUG
    if (DEBUG) {
        int ix = 0;
        for (auto &trk_i: objects) {
            if (trk_i->isActive()) {
                trk_i->drawTrack(draw_img, ix);
                //string speed_text = "Speed: " + to_string(trk_i->getSpeed()) + " px/frame";
                //Point speed_point = trk_i->getCurrentPosition();
                //putText(draw_img, speed_text, speed_point, FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0,255,0), 2);
            }
            Scalar TXT_COLOR = EB_BLU;
            string NUMDET = "#D: " + to_string(_detections.size());
            string NUMTRK = "#T: " + to_string(getActiveTrackers());
            Point pt0(10, 60);
            Point pt1(10, 90);
            putText(draw_img, NUMDET , pt0, FONT_HERSHEY_SIMPLEX, 0.75, TXT_COLOR, 2);
            putText(draw_img, NUMTRK , pt1, FONT_HERSHEY_SIMPLEX, 0.75, TXT_COLOR, 2);
        }
    }
}

Mat CentroidTracker::getDrawImage() {
	return draw_img;
}
int TrackingObject::getSpeed() {
	return speed_kmh;
}
Mat CentroidTracker::getDetsImage() {
	return dets_img;
}

void CentroidTracker::evalObjects() {
	for (auto &trk_i: objects) {
		if (trk_i->isActive()) {
			bool dead_object = trk_i->lifeControl();
			if (dead_object) {
				trk_i->setToDelete(dead_object);
				//cout << "*-*-*-*-*-*-* Delete tracker : "<< trk_i->getId()<<endl;
			}
		}
	}
}

/*
 @ Search in objects to delete and check if it is ready to report.
 * */
Json::Value CentroidTracker::reportObjects(bool &save_img) {
	save_img = false;
	Json::Value trackers = Json::Value(Json::arrayValue);
	int idx = 0;
	for (auto &trk_i: objects) {
		if (trk_i->isActive()) {
			bool to_send = false;
			float FX = (float)1.0/frame.cols;
			float FY = (float)1.0/frame.rows;            

			Json::Value partInfo;
			dnn_bbox obj_tail = trk_i->getRouteBboxTail();          
			partInfo["photo_object_cutted"] = obj_tail.photo_object_cutted;
			partInfo["prob_det"] = obj_tail.prob_det;
			partInfo["embeddings"] = obj_tail.embeddings;
			partInfo["obj_id"]  = obj_tail.obj_id;
			partInfo["tag"] = obj_tail.label;
			partInfo["uuid"] = obj_tail.uuid;
			partInfo["tracker_id"] = trk_i->getId();
			partInfo["init_time"] = trk_i->getTimestampIn();
			partInfo["duration_time"] = trk_i->getDuration();
			partInfo["tracker_size"] = trk_i->getRouteSize();
			partInfo["state"] = trk_i->getStatus();            
			partInfo["init_frame_id"] = trk_i->getInitFrameId();
			//partInfo["speed"] = trk_i->getSpeed();
			//cout<<trk_i->getSpeed()<<"  speed   ***************"<<endl;
			if (trk_i->isAwake()) {
				Rect ri = obj_tail.bbox;
				partInfo["bbox"]["x"] = to_string_with_precision(ri.x * FX, 4);
				partInfo["bbox"]["y"] = to_string_with_precision(ri.y * FY, 4);
				partInfo["bbox"]["w"] = to_string_with_precision(ri.width * FX, 4);
				partInfo["bbox"]["h"] = to_string_with_precision(ri.height * FY, 4);
				partInfo["prob"]      = to_string_with_precision(obj_tail.prob, 4);
				to_send = true;
			}

			Json::Value jpoligon_inside = Json::Value(Json::arrayValue);
			for(int i=0; i<trk_i->polygon_inside.size();i++){
				jpoligon_inside[i] = trk_i->polygon_inside[i];
			}
			partInfo["speed"] = trk_i->getSpeed();
			partInfo["poligon_inside"] = jpoligon_inside;
			partInfo["poligon_events"] = trk_i->polygon_events;
			//cout<<trk_i->polygon_events<<endl;
			if(trk_i->getStatus() == 0){save_img = true;}

			if (trk_i->isToDelete()) {
				partInfo["end_time"] = trk_i->getTimestampOut();
				to_send = true;
			}

			if (to_send) {
				vector<Point> route_to_send = trk_i->getRouteToSend();
				for (int i=0; i<route_to_send.size();i++) {
					string xval = "x"+to_string(i+1);
					string yval = "y"+to_string(i+1);
					partInfo["route"][xval] = to_string_with_precision(route_to_send[i].x*FX, 4);
					partInfo["route"][yval] = to_string_with_precision(route_to_send[i].y*FY, 4);
				}
				trackers[idx] = partInfo;
				idx++;
				trk_i->incNumSent();
			}

		}
	}
	return trackers;
}

/*
 @ Check and shut down vehicles .
 * */
void CentroidTracker::deactivateObjects() {
	for (auto &trk_i: objects) {
		if (trk_i->isToDelete())
			trk_i->shutDown();
	}
}

int CentroidTracker::getActiveTrackers() {
	int count = 0;
	for (auto &trk_i: objects) {
		if (trk_i->isActive())
			count++;
	}
	return count++;
}


void CentroidTracker::setPoligonForSpeed(const Json::Value& poligon)
{

	std::vector<Json::Value> poligon_objects = get_list_of_json(poligon);
	// se redimensiona el tracker para almacenar todos los poligonos

	
	int index_pol = 0;
	for (size_t i = 0, n = poligon_objects.size(); i < n; ++i) {
		// obtienen el id del tracker
		bool check_id = poligon_objects[i].isMember("poligon_id");
		if (!check_id) {
			std::cerr << "ERROR: this poligon does not have an \"poligon_id\" key\n";
			continue;
		}
		std::string poligon_id = poligon_objects[i]["poligon_id"].asString();
		cout<<poligon_id<<"  points    *************"<<endl;
		// obtiene los puntos del tracker en formato json
		bool check_id_0 = poligon_objects[i].isMember("points");
		if (!check_id_0) {
			std::cerr << "ERROR: wrong input on poligon id: " << poligon_id << "\n";
			std::cerr << "ERROR: this poligon does not have an \"points\" key\n";
			continue;
		}
		bool check_id_1 = poligon_objects[i].isMember("meters");
		if (!check_id_1) {
			std::cerr << "ERROR: wrong input on poligon id: " << poligon_id << "\n";
			std::cerr << "ERROR: this poligon does not have a \"meters\" key\n";
			continue;
		}
				////////////////////////////////////////
		
		std::vector<Json::Value> poligon_points = get_list_of_json(poligon_objects[i]["points"]);
		
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
			cout<<x<<"   "<<y<<" y   *********************"<<endl;
			speed_poligon.push_back(cv::Point2f(x,y));
		}
		////////////////////////////////////////
		
		
		std::vector<Json::Value> poligon_meters = get_list_of_json(poligon_objects[i]["meters"]);
		
		bool temp_flag_meters = false;
		size_t m_1 = poligon_meters.size();
		if (m < 3) {
			std::cerr << "ERROR: wrong input on poligon id: " << poligon_id << "\n";
			std::cerr << "ERROR: number of poligon is lower to 3" << std::endl;
			continue;
		}
		for (size_t j = 0; j < m_1; ++j) {
			// se leen las coordenadas del punto
			bool check_x = poligon_meters[j].isMember("x");
			if (!check_x) {
				std::cerr << "ERROR: wrong input on poligon id: " << poligon_id << "\n";
				std::cerr << "ERROR: point " << (j + 1) << " does no have attribute \"x\"" << std::endl;
				temp_flag_meters = true;
				break;
			}
			float x_1 = poligon_meters[j]["x"].asFloat();
			bool check_y_1 = poligon_points[j].isMember("y");
			if (!check_y_1) {
				std::cerr << "ERROR: wrong input on poligon id: " << poligon_id << "\n";
				std::cerr << "ERROR: point " << (j + 1) << " does no have attribute \"y\"" << std::endl;
				temp_flag_meters = true;
				break;
			}
			float y_1 = poligon_meters[j]["y"].asFloat();
			//cout<<x_1<<"   "<<y_1<<" y   *********************"<<endl;
			speed_meters.push_back(cv::Point2f(x_1,y_1));
		}
		
	}

}

