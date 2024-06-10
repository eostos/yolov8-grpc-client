#include "poligonTrackerManager.hpp"


Json::Value PoligonTrackerManager::evaluate() {

    if (this->_eval_points.size() == 0) {
        size_t n = this->_points.size();
        assert(n > 2);
        cv::Point2f suma_puntos;
        for (size_t i = 0; i < n; ++i) {
            cv::Point2f pf;
            pf.x = this->_points[i].x * this->_size.width;
            pf.y = this->_points[i].y * this->_size.height;
            this->_eval_points.push_back(pf);
            suma_puntos.x += pf.x;
            suma_puntos.y += pf.y;
        }
        this->_center_point.x = suma_puntos.x / n;
        this->_center_point.y = suma_puntos.y / n;

        this->_poligon_area = contourArea(_eval_points);
    }

    // bool flag = false;

    size_t number_tracker = this->_trackers.size();

    std::vector<std::string> trackers_inside = {};
    std::vector<PoligonEvent> poligon_event = {};
    // std::vector <>

    for (size_t i = 0; i < number_tracker; ++i) {
        TrackingObject *to = this->_trackers[i];

        // si el tracker NO esta activo se ignora
        if (!to->isActive()) { continue; }

        // ignora los tracker para objectos diferentes
        // if (to.getObjId() != this->getObjectID()) {
        //     continue;
        // }

        // se obtiene la ruta y su tamano, si el tamano es menor a dos
        // no se puede determinar si hay entrada o salida.
        std::vector<cv::Point> route = to->getRoute();
        std::vector<dnn_bbox> route_bbox = to->getBboxRoute();
        size_t route_size = route.size();
        size_t route_bbox_size = route_bbox.size();
        if (route_size < 1) {
            continue;
        }

        // se obtien el ultimo punto y se determina si el ultimo punto esta
        // dentro del poligono
        cv::Point p1 = route[route_size - 1];

        /*float status_1 = jsignus(cv::pointPolygonTest(this->_eval_points, p1, true));

        // se toman los estados "BORDE" como "DENTRO"
        if (status_1 == 0) { status_1 = 1; }
        if (status_1 == 1) {
            trackers_inside.push_back(to->getId());
        }*/

        dnn_bbox dnn_bbox1 = route_bbox[route_bbox_size-1];        
        Rect bbox1 = dnn_bbox1.bbox;

        cv::Point p1_1 = cv::Point(bbox1.x,bbox1.y);
        cv::Point p1_2 = cv::Point(bbox1.x+bbox1.width,bbox1.y);
        cv::Point p1_3 = cv::Point(bbox1.x+bbox1.width,bbox1.y+bbox1.height);
        cv::Point p1_4 = cv::Point(bbox1.x,bbox1.y+bbox1.height);

        vector<Point> p1_to_eval {p1_1,p1_2,p1_3,p1_4};
        float status_1 = false;
        for (auto p_to_eval: p1_to_eval){
            float inside = jsignus(cv::pointPolygonTest(this->_eval_points, p_to_eval, true));       
            if (inside==0 || inside==1){ // se toman los estados "BORDE" como "DENTRO"
                status_1 = true;
                trackers_inside.push_back(to->getId());
                break;
            }
        }

        if (route_size < 2) {
            continue;
        }
        // se obtienen los dos ultimos elementos de la ruta del tracker
        cv::Point p0 = route[route_size - 2];

        // se determinae el estado "DENTRO" "FUERA" o "BORDE" de cada uno de los puntos
        // dentro = 1
        // borde = 0
        // fuera = -1
        float status_0 = jsignus(cv::pointPolygonTest(this->_eval_points, p0, true));

        // si el tracker se mantiene dentro del poligono
        if (status_0 == 1 && status_1 == 1) { continue; }

        // se toman los estados "BORDE" como "DENTRO"
        if (status_0 == 0) { status_0 = 1; }

        std::string type_event;

        bool flag = false;

        if ((status_0 == -1 && status_1 == 1)) {
            // std::cout << "IN" << std::endl;
            flag = true;
            type_event = "IN";
        } 

        if ((status_0 == 1 && status_1 == -1)) {
            // std::cout << "OUT" << std::endl;
            flag = true;
            type_event = "OUT";
        }

        int lado = 0;
        cv::Point2f inter_point;
        if (flag) {
            size_t n_points = this->_eval_points.size();
            for (size_t j = 0; j < n_points; ++j) {
                cv::Point2f p2 = _eval_points[j];
                cv::Point2f p3 = _eval_points[(j + 1) % n_points];
                
                // inter_point = interceptionPoint(p0, p1, p2, p3);   
                // bool new_flag  = checkPoint(p2, p3, inter_point);
                // if (new_flag) {
                //     std::cout << "LADO " << j << std::endl;
                //     lado = j;
                //     break;
                // }
                // std::cout << "trying lado: " << j << std::endl;
                cv::Point2f pout;
                // puntos del lado, luego puntos del tracker, por ultimo, 
                // punto de salida
                bool k = intermediate_point(p2, p3, p0, p1, pout);
                if (k) {
                    inter_point = pout;
                    lado = j;
                    // std::cout << "lado: " << j << std::endl;
                    // sstd::cout << "lado: " << j << std::endl;
                    // std::this_thread::sleep_for(std::chrono::seconds(300));
                    // throw;
                    break;
                }
            }
        }

        // EVENTO DE CRUCE
        if (status_0 == -1 && status_1 == -1) {
            // std::cout << "doble salida" << std::endl;
            // ----------------------------------------------------------------
            // se detrmina si hubo una posible entrada por alguno de los lados
            // ----------------------------------------------------------------
            size_t n_points = this->_eval_points.size(); 
            for (size_t j = 0; j < n_points; ++j) {
                // std::cout << "primer loop: " << j << std::endl;
                cv::Point2f p2 = _eval_points[j];
                cv::Point2f p3 = _eval_points[(j + 1) % n_points];
                cv::Point2f pout;                
                bool flag0 = intermediate_point(p2, p3, p0, p1, pout);
                // en caso que no encuentre un punto de cruce por alguno de los
                // lados, se ignora
                if (!flag0) {
                    continue;
                }
                // std::cout << "flag0: TRUE\n";
                // implicitamente de aqui en adelante flag0 es True

                // // se halla el angulo entre el vector desplazamiento y el
                // // el punto de origin del despezamiento y el centro del poligono
                // cv::Point2f d0 = cv::Point2f(p1.x - p0.x, p1.y - p0.y);
                // cv::Point2f pc = cv::Point2f(_center_point.x - p0.x, _center_point.y - p0.y);
                // double cos_angle0 = (d0.x * pc.x + pc.y * d0.y) / (cv::norm(pc) * cv::norm(d0));
                // // std::cout << "cos_angle0: " << cos_angle0 << std::endl;

                cv::Point2f pc = this->_center_point;
                cv::Point2f d2 = cv::Point2f(p2.x - pc.x, p2.y - pc.y);
                cv::Point2f d3 = cv::Point2f(p3.x - pc.x, p3.y - pc.y);
                cv::Point2f d23 = cv::Point2f(d2 + d3);

                cv::Point2f d0 = cv::Point2f(pc.x - p0.x, pc.y - p0.y);

                double den = cv::norm(d0) * cv::norm(d23);
                // double cos_angle0 = (d0.x * pc.x + pc.y * d0.y) / (cv::norm(pc) * cv::norm(d0));
                // std::cout << "cos_angle0: " << cos_angle0 << std::endl;

                // si el cosano del angulo es positivo es una evento de entrada
                // en caso contrario, es de salida
                // bool in_flag = cos_angle0 > 0;

                bool in_flag = false;
                if (den != 0) {
                    // se halla la proyeccion de un verctor sobre el otro: producto punto
                    double cos_angle0 = (d0.x * d23.x + d0.y * d23.y) / den;
                    in_flag = cos_angle0 > 0;
                } else {
                    // se halla el producto cruz vectorial
                    float k = d23.x * d0.y - d23.y * d0.x;
                    in_flag = k > 0;
                }

                // ----------------------------------------------------------------
                // se busca otro punto de cruce, pero esta vez por los lados que 
                // faltan
                for (size_t k = j + 1; k < n_points; ++k) {
                    // std::cout << "segunda iteracion: " << k << std::endl;
                    cv::Point2f p4 = _eval_points[k];
                    cv::Point2f p5 = _eval_points[(k + 1) % n_points];
                    cv::Point2f pout2;
                    // 
                    bool flag1 = intermediate_point(p4, p5, p0, p1, pout2);
                    // std::cout << "flag1: " << flag1 << std::endl;
                    // si se encontro otro punto de cruce
                    if (flag1) {

                        // CREAMOS EL EVENTO IN
                        // std::cout << "pin\n";
                        PoligonEvent pe;
                        pe.lado = j;
                        pe.point = pout;
                        pe.type = in_flag ? "IN" : "OUT";
                        pe.tracker_id = to->getId();
                        pe.id_object = to->getObjId();

                        // std::cout << "push\n"; 
                        poligon_event.push_back(pe);

                        // CREAMOS EL EVENTO OUT
                        // std::cout << "pout\n";
                        PoligonEvent pe2;
                        pe2.lado = k;
                        pe2.point = pout2;
                        pe2.type = in_flag ? "OUT" : "IN";
                        pe2.tracker_id = to->getId();
                        pe2.id_object = to->getObjId();

                        // std::cout << "push\n"; 
                        poligon_event.push_back(pe2);


                        break;
                    }
                }
                // ----------------------------------------------------------------
            }
            // ----------------------------------------------------------------
        }

        // se agregan los eventos de entrada y salida
        if (flag) {
            PoligonEvent pe;
            pe.lado = lado;
            pe.point = inter_point;
            pe.type = type_event;
            pe.tracker_id = to->getId();
            pe.id_object = to->getObjId();
            poligon_event.push_back(pe);
        }
    
    }

    Json::Value json_result;
    // json_result["trackers_inside"] = 

    Json::Value json_trackers_inside = Json::Value(Json::arrayValue);
    for (int i = 0, n = (int)trackers_inside.size(); i < n; ++i) {
        json_trackers_inside[i] = trackers_inside[i];
    }

    Json::Value poligon_events = Json::Value(Json::arrayValue);
    for (int i = 0, n = (int)poligon_event.size(); i < n; ++i) {
        Json::Value event_poligon;
        PoligonEvent pe = poligon_event[i];
        //
        Json::Value temp_point;
        // se normaliza el punto de intercepcion con respecto a las dimensiones de la imagen
        temp_point["x"] = pe.point.x / static_cast<float>(this->getSize().width);
        temp_point["y"] = pe.point.y / static_cast<float>(this->getSize().height);
        //
        event_poligon["type_event"] = pe.type;
        event_poligon["side"] = pe.lado;
        event_poligon["interception_point"] = temp_point;
        event_poligon["tracker_id"] = pe.tracker_id;
        event_poligon["object_id"] = pe.id_object;
        event_poligon["poligon_id"] = this->getPoligonID();
        
        poligon_events[i] = event_poligon;
    }

    json_result["poligon_id"] = this->getPoligonID();
    json_result["trackers_inside"] = json_trackers_inside;
    json_result["poligon_events"] = poligon_events;

    //cout << poligon_events <<endl;
    //cout << "----------------------------------------------------------------------------------" <<endl;

    return json_result;
}


float jsignus(float f) {
    if (f < 0) { return -1; }
    if (f > 0) { return 1; }
    return 0;
}


cv::Point2f interceptionPoint(cv::Point2f p0, cv::Point2f p1, cv::Point2f p2, cv::Point2f p3) {


    // std::cout << "p0: " << p0 <<  " p1 " << p1 << std::endl;
    // std::cout << "p2: " << p2 <<  " p3 " << p3 << std::endl;
    // assert(p0.x != p1.x);
    assert(p0.x != p1.x || p0.y != p1.y);
    // assert(p2.x != p3.x);
    assert(p2.x != p3.x || p2.y != p3.y);
    //
    float numx = (p2.y - p0.y) * (p3.x - p2.x) * (p1.x - p0.x) + (p1.y - p0.y) * (p3.x - p2.x) * p0.x - (p3.y - p2.y) * (p1.x - p0.x) * p2.x;
    float denx = (p1.y - p0.y) * (p3.x - p2.x) - (p3.y - p2.y) * (p1.x - p0.x);
    float x = numx/denx;
    //
    float y = (p1.y - p0.y) * (x - p0.x) / (p1.x - p0.x) + p0.y;
    return cv::Point2f(x,y);
}


bool checkPoint(cv::Point2f p0, cv::Point2f p1, cv::Point2f p2) {
    assert(p0.x != p1.x || p1.y != p0.y);
    // se comprueba el punto pertenece al segmento
    float ep = 1e-4;
    float y0 = (p0.y - p1.y) * (p2.x - p0.x)/ (p0.x - p1.x) + p0.y;
    // se utiliza un epsilon: error permitido
    bool cond1 = abs(y0 - p2.y) < ep;
    if (!cond1) { return false; }
    // se determina si x esta en el intervalo del segmento
    if (p0.x != p1.x) {
        float minx = min(p0.x, p1.x);
        float maxx = max(p0.x, p1.x);
        if (p2.x >= minx && p2.x <= maxx) { return true; }    
    } else {
        float minx = min(p0.y, p1.y);
        float maxx = max(p0.y, p1.y);
        if (p2.y >= minx && p2.y <= maxx) { return true; }    
    }
    return false;
}


bool intermediate_point(cv::Point2f p0, cv::Point2f p1,
        cv::Point2f p2, cv::Point2f p3, cv::Point2f &pc) {
    // // --------------------------------------------------------------
    // // ALGORITMO DE PUNTO INTERMEDIO
    // // --------------------------------------------------------------
    // std::cout << "P0: " << p0 << std::endl;
    // std::cout << "P1: " << p1 << std::endl;
    // std::cout << "P2: " << p2 << std::endl;
    // std::cout << "P3: " << p3 << std::endl;
    // float a11 = (p0.y - p1.y);
    // float a12 = (p1.x - p0.x);
    // float a21 = (p2.y - p3.y);
    // float a22 = (p3.x - p2.x);
    // float det0 = a11 * a22 - a21 * a12;
    // if (det0 == 0) { return false; }
    // float b1 = p0.x * p1.y + p1.x * p0.y;
    // float b2 = p2.x * p3.y + p3.x * p2.y;
    // float Mx = b1 * a22 - b2 * a12;
    // float My = a11 * b2 - a21 * b1;
    // pc.x = Mx / det0;
    // pc.y = My / det0;
    // std::cout << "Pout: " << pc << std::endl;
    // // --------------------------------------------------------------

    // assert(p1.y != p0.y || p1.x != p0.x);
    // assert(p2.y != p3.y || p2.x != p3.x);

    if (p1.y == p0.y && p1.x == p0.x) { return false; }
    if (p3.y == p2.y && p2.x == p3.x) { return false; }


    // // --------------------------------------------------------------

    // determinamos los maximos y minimos del segmento en x
    float minx_0 = min(p0.x, p1.x);
    float maxx_0 = max(p0.x, p1.x);
    // determinamos los maximos y los minimos del tracker en x
    float minx_2 = min(p2.x, p3.x);
    float maxx_2 = max(p2.x, p3.x);

    if (maxx_0 < minx_2) { return false; }
    if (maxx_2 < minx_0) { return false; }

    // determinamos los maximos y minimos del segmento en y
    float miny_0 = min(p0.y, p1.y);
    float maxy_0 = max(p0.y, p1.y);
    // determinamos los maximos y los minimos del tracker en y
    float miny_2 = min(p2.y, p3.y);
    float maxy_2 = max(p2.y, p3.y);

    if (maxy_0 < miny_2) { return false; }
    if (maxy_2 < miny_0) { return false; }

    // // --------------------------------------------------------------
    // // OTRO ALGORIMO
    // // --------------------------------------------------------------
    // se calculan los cambios en el lado
    float delta_y_0 = p1.y - p0.y;
    float delta_x_0 = p1.x - p0.x;

    // se calculas los cambios en el tracker
    float delta_y_2 = p3.y - p2.y;
    float delta_x_2 = p3.x - p2.x;

    // --------------------------------------------------
    // en caso que lado sea vertical y el tracker sea vertical tambien
    // se ignora
    // --------------------------------------------------
    if (delta_x_2 == 0 && delta_x_0 == 0) { 
        return false;
    }
    // --------------------------------------------------


    // --------------------------------------------------
    // en caso que lado sea vertical
    // --------------------------------------------------
    if (delta_x_0 == 0) {
        // se calcula la pendiente del tracker
        float m2 = delta_y_2 / delta_x_2;
        // se aplica la ecuacon de la recta para encontrar
        // el punto intermedio
        // std::cout << "m2: " << m2 << std::endl;
        float y_c = m2 * (p0.x - p2.x) + p2.y; 
        // std::cout << "y_c: " << y_c << std::endl;
        //
        // se determina si el punto se intercepcion se
        // encuentra en el segmento del lado
        float miny = min(p0.y, p1.y);
        float maxy = max(p0.y, p1.y);
        if (y_c >= miny && y_c <= maxy) {
            pc.x = p0.x;
            pc.y = y_c;
            // std::cout << "Pout: " << pc << std::endl;
            return true;
        }
        //
        // el punto de intercepcion no se encuentra en el
        // segmento del lado
        return false;
    }
    // --------------------------------------------------

    // --------------------------------------------------
    // en caso que el tracker sea  vertical
    // --------------------------------------------------
    if (delta_x_2 == 0) {
        // se calcula la pendiente del lado
        float m0 = delta_y_0 / delta_x_0;
        // se aplica la ecuacion de la recta para encontrar
        // el punto de intercepsion
        // std::cout << "m0: " << m0 << std::endl;
        float y_c = m0 * (p2.x - p0.x) + p0.y;
        // std::cout << "y_c: " << y_c << std::endl;
        // throw;

        //
        // se determina si el punto se intercepcion se
        // encuentra en el segmento del tracker
        float miny = min(p2.y, p3.y);
        float maxy = max(p2.y, p3.y);
        if (y_c >= miny && y_c <= maxy) {
            pc.x = p2.x;
            pc.y = y_c;
            // std::cout << "Pout: " << pc << std::endl;
            return true;
        }
        //
        // el punto de intercepcion no se encuentra en el
        // segmento del lado
        // std::cout << "FAKE" << std::endl;
        // throw;

        return false;
    }
    // --------------------------------------------------


    // throw;

    // se calculan las pendientes
    float m0 = delta_y_0 / delta_x_0;
    float m2 = delta_y_2 / delta_x_2;

    // --------------------------------------------------
    // en caso que las pendientes den como resultado cero
    // tambien se ignora
    // --------------------------------------------------
    // std::cout << "m0: " << m0 << std::endl;
    // std::cout << "m2: " << m2 << std::endl;

    if (m0 == 0 && m2 == 0) { return false; }
    // --------------------------------------------------

    // --------------------------------------------------
    // en caso  que tengan la misma pendiente
    // se ignora tambien
    // --------------------------------------------------
    if (m0 == m2) { return false; }
    // --------------------------------------------------

    // --------------------------------------------------
    // se halla los valores de los coeficientes
    // --------------------------------------------------
    float a12 = 1;
    float a22 = 1;
    float a11 = -1 * m0;
    float a21 = -1 * m2;
    // --------------------------------------------------
    // se halla el valor del determinate
    // --------------------------------------------------
    float det0 = a11 * a22 - a21 * a12;
    // std::cout << "det0: " << det0 << std::endl;
    if (det0 == 0) { return false; }
    // b1 y b2
    float b1 = p0.y - m0 * p0.x;
    float b2 = p2.y - m2 * p2.x;
    // se hallan los otros determinantes
    float Mx = b1 * a22 - b2 * a12;
    float My = a11 * b2 - a21 * b1;
    pc.x = Mx / det0;
    pc.y = My / det0;


    // // --------------------------------------------------------------
    // std::cout << "Pout: " << pc << std::endl;
    // // --------------------------------------------------------------
    return true;
}

// bool check_crossing_event(cv::Point2f p0, cv::Point2f p1, cv::Point2f pc) {
//     cv::Point2f d0 = cv::Point2f(pc.x - p0.x, pc.y - p0.y);
//     cv::Point2f d1 = cv::Point2f(p1.x - pc.x, p1.y - pc.y);
//     double angle = (d0.x * d1.x + d0.y * d1.y) / cv::norm(d0 - d1);
// }

// sbool check_crossing_event(cv::Point2f p0, cv::Point2f p1, cv)

Json::Value LineTrackerManager::evaluate() {

    if (this->_eval_points.size() == 0) {
        size_t n = this->_points.size();
        if (n != 2) { std::cerr << "ERROR, NUMBER OF POINTS IS DIFERENT TO 2\n"; }
        assert(n == 2);
        // cv::Point2f suma_puntos;
        for (size_t i = 0; i < n; ++i) {
            cv::Point2f pf;
            pf.x = this->_points[i].x * this->_size.width;
            pf.y = this->_points[i].y * this->_size.height;
            this->_eval_points.push_back(pf);
            // suma_puntos.x += pf.x;
            // suma_puntos.y += pf.y;
        }
        // this->_center_point.x = suma_puntos.x / n;
        // this->_center_point.y = suma_puntos.y / n;
    }

    // bool flag = false;

    size_t number_tracker = this->_trackers.size();

    std::vector<std::string> trackers_inside = {};
    std::vector<PoligonEvent> poligon_event = {};
    // std::vector <>

    for (size_t i = 0; i < number_tracker; ++i) {
        TrackingObject *to = this->_trackers[i];

        // si el tracker NO esta activo se ignora
        if (!to->isActive()) { continue; }

        // ignora los tracker para objectos diferentes
        // if (to.getObjId() != this->getObjectID()) {
        //     continue;
        // }

        // se obtiene la ruta y su tamano, si el tamano es menor a dos
        // no se puede determinar si hay entrada o salida.
        std::vector<cv::Point> route = to->getRoute();
        size_t route_size = route.size();
        if (route_size < 2) {
            continue;
        }

        // se obtien el ultimo punto y se determina si el ultimo punto esta
        // dentro del poligono
        cv::Point p1 = route[route_size - 1];
        cv::Point p0 = route[route_size - 2];

        cv::Point p2 = this->_eval_points[0];
        cv::Point p3 = this->_eval_points[1];

        cv::Point2f d0 = cv::Point(p1.x - p0.x, p1.y - p0.y);
        cv::Point2f d23 = cv::Point(p3.x - p2.x, p3.y - p2.y);

        // std::cout << "trying lado: " << j << std::endl;
        cv::Point2f pout;
        // puntos del lado, luego puntos del tracker, por ultimo, 
        // punto de salida
        bool flag = intermediate_point(p2, p3, p0, p1, pout);
        cout << "Flag intermediate_point  " << flag << endl;

        if (flag) {

            // producto cruz
            double kv = d23.x * d0.y - d23.y * d0.x;  
            std::string type_event = kv > 0 ? "IN" : "OUT";
            cout << "Type Event  " << type_event << endl;
            // se agregan los eventos de entrada y salida
            PoligonEvent pe;
            pe.lado = 0;
            pe.point = pout;
            pe.type = type_event;
            pe.tracker_id = to->getId();
            pe.id_object = to->getObjId();
            poligon_event.push_back(pe);
        }
    
    }

    Json::Value json_result;
    // json_result["trackers_inside"] = 

    Json::Value json_trackers_inside = Json::Value(Json::arrayValue);

    Json::Value poligon_events = Json::Value(Json::arrayValue);
    for (int i = 0, n = (int)poligon_event.size(); i < n; ++i) {
        Json::Value event_poligon;
        PoligonEvent pe = poligon_event[i];
        //
        Json::Value temp_point;
        // se normaliza el punto de intercepcion con respecto a las dimensiones de la imagen
        temp_point["x"] = pe.point.x / static_cast<float>(this->getSize().width);
        temp_point["y"] = pe.point.y / static_cast<float>(this->getSize().height);
        //
        event_poligon["type_event"] = pe.type;
        event_poligon["side"] = pe.lado;
        event_poligon["interception_point"] = temp_point;
        event_poligon["tracker_id"] = pe.tracker_id;
        event_poligon["poligon_id"] = this->getPoligonID();
        event_poligon["object_id"] = pe.id_object;
        
        poligon_events[i] = event_poligon;
    }

    json_result["poligon_id"] = this->getPoligonID();
    json_result["trackers_inside"] = json_trackers_inside;
    json_result["poligon_events"] = poligon_events;

    return json_result;
}

bool calculateIntersection(Point A, Point B, Point C, Point D, Point2f &out){
    // Line p0p1 represented as a1x + b1y = c1
    double a1 = B.y - A.y;
    double b1 = A.x - B.x;
    double c1 = a1*(A.x) + b1*(A.y);
 
    // Line p2p3 represented as a2x + b2y = c2
    double a2 = D.y - C.y;
    double b2 = C.x - D.x;
    double c2 = a2*(C.x)+ b2*(C.y);
 
    double determinant = a1*b2 - a2*b1;

    if(determinant==0){
        return false;
    }else{
        double x = (b2*c1 - b1*c2)/determinant;
        double y = (a1*c2 - a2*c1)/determinant;
        bool C1 = checkPoint(C, D, cv::Point2f(x,y));
        bool C2 = checkPoint(A, B, cv::Point2f(x,y));
        if(C1&&C2){
            out.x = x;
            out.y = y;
            return true;        
        }else{
            return false;
        }


    }

}

void PoligonTrackerManager::evaluateAreaBbox() {

    if (this->_eval_points.size() == 0) {
        size_t n = this->_points.size();
        assert(n > 2);
        cv::Point2f suma_puntos;
        for (size_t i = 0; i < n; ++i) {
            cv::Point2f pf;
            pf.x = this->_points[i].x * this->_size.width;
            pf.y = this->_points[i].y * this->_size.height;
            this->_eval_points.push_back(pf);
            suma_puntos.x += pf.x;
            suma_puntos.y += pf.y;
        }
        this->_center_point.x = suma_puntos.x / n;
        this->_center_point.y = suma_puntos.y / n;
        this->_poligon_area = contourArea(_eval_points);
    }

    size_t number_tracker = this->_trackers.size();
    for (size_t i = 0; i < number_tracker; ++i) {
        TrackingObject *to = this->_trackers[i];

        // si el tracker NO esta activo se ignora
        if (!to->isActive()) { continue; }
        
        std::vector<float> route_area = to->getBboxArea();//return area of each detection bbox vector
        size_t route_area_size = route_area.size();

        if (route_area_size < 1) {
            continue;
        }

        vector<Point> intersection_polygon1;
        vector<vector<Point2f>> route_poligon = to->getBboxPoligon();
        //cout <<this->getPoligonID()<< " ID , this->_eval_points " << this->_eval_points<< " route_poligon.back() " << route_poligon.back()  << "  end_2"<<endl;
        //Compara el poligono que esta pisando la punta de tracker con el 
        //this->_eval_points = esto es el poligono el cual se esta evaluando , guardado en el config
        //route_poligon.back() = rect tipo poligono , bounding box , ultimo 
        float intersect_area1 = intersectConvexConvex(this->_eval_points, route_poligon.back(), intersection_polygon1, true);
        //cout << "Points Intersected " << intersect_area1<< " Poligon ID " << this->getPoligonID()  << endl;
        //cout <<route_poligon.size()<< "size "<< "BBOX LAST" << route_poligon.back() << " BBOX FIRST  " << route_poligon[0]  << endl;
        float perc1 = intersect_area1/(route_area.back());
        float status_1 = false;
        //cout <<perc1<< " PORCENTAGE AREA :  "  <<endl;
        if(perc1>0.30){
            to->setPolygon(this->getPoligonID());
            status_1 = true;
        }

        if (route_area_size < 6 || !to->isAwake()) {
            continue;
        }

        vector<Point> intersection_polygon0;
        float intersect_area0 = intersectConvexConvex(this->_eval_points, route_poligon[route_area_size-6], intersection_polygon0, true);
        float perc0 = intersect_area0/(route_area[route_area_size-6]);
        float status_0 = false;
        
        if(perc0>0.30){
            status_0 = true;
        }

        // si el tracker se mantiene dentro del poligono
        if (status_0 == 1 && status_1 == 1) { continue; }

        std::string type_event;
        bool flag = false;

        if ((status_0 == 0 && status_1 == 1)) {
            flag = true;
            type_event = "IN";

            
        } 

        if ((status_0 == 1 && status_1 == 0)) {
            flag = true;
            type_event = "OUT";
        } 

        if (flag) {
            vector<Point> route_center = to->getRoute();
            size_t route_center_size = route_center.size();

            Point p1 = route_center[route_center_size-1];
            Point p0 = route_center[route_center_size-2];     

            Point p0_ext;
            p0_ext.x = 1.5*p0.x - 0.5*p1.x;
            p0_ext.y = 1.5*p0.y - 0.5*p1.y;

            Point p1_ext;
            p1_ext.x = 1.5*p1.x - 0.5*p0.x;
            p1_ext.y = 1.5*p1.y - 0.5*p0.y;

            int lado = -1;
            cv::Point2f inter_point;     

            size_t n_points = this->_eval_points.size();
            for (size_t j = 0; j < n_points; ++j) {
                cv::Point2f p2 = _eval_points[j];
                cv::Point2f p3 = _eval_points[(j + 1) % n_points];
                cv::Point2f pout;
                bool k = calculateIntersection(p0_ext,p1_ext,p2,p3,pout);
                if(k){
                    lado = j;
                    Json::Value event_poligon;
                    event_poligon["type_event"] = type_event;
                    event_poligon["side"] = lado;
                    if(this->getPoligonID()=="17"){
                    cout <<this->getPoligonID()<< " ID , this->_eval_points   , LADO :  " << lado <<" TYPE " <<type_event<<endl;
                    }
                    Json::Value temp_point;
                    // se normaliza el punto de intercepcion con respecto a las dimensiones de la imagen
                    temp_point["x"] = pout.x / static_cast<float>(this->getSize().width);
                    temp_point["y"] = pout.y / static_cast<float>(this->getSize().height);
                    //
                    event_poligon["interception_point"] = temp_point;
                    event_poligon["poligon_id"] = this->getPoligonID();
                    to->setPolygonEvent(event_poligon);
                    
                }
            }
        }               
    }
}
void PoligonTrackerManager::evaluateArea() {
    //first points 
    if (this->_eval_points.size() == 0) {
        size_t n = this->_points.size();
        assert(n > 2);
        cv::Point2f suma_puntos;
        for (size_t i = 0; i < n; ++i) {
            cv::Point2f pf;
            pf.x = this->_points[i].x * this->_size.width;
            pf.y = this->_points[i].y * this->_size.height;
            this->_eval_points.push_back(pf);
            suma_puntos.x += pf.x;
            suma_puntos.y += pf.y;
        }
        this->_center_point.x = suma_puntos.x / n;
        this->_center_point.y = suma_puntos.y / n;
        this->_poligon_area = contourArea(_eval_points);
    }

    size_t number_tracker = this->_trackers.size();
    //number of trackers
    for (size_t i = 0; i < number_tracker; ++i) {
        TrackingObject *to = this->_trackers[i];

        // si el tracker NO esta activo se ignora
        if (!to->isActive()) { continue; }
        
        std::vector<float> route_area = to->getBboxArea();//return area of each detection bbox vector
        size_t route_area_size = route_area.size();

        if (route_area_size < 6) {//VERY IMPORTANT TO ADD THE FPS VARIABLE AND CHANGE THIS AUTOMATICLLY AND NOT MANYALLY 
            continue;
        }

        vector<Point> intersection_polygon1;
        vector<vector<Point2f>> route_poligon = to->getBboxPoligon();
        //vector<Point> route_center = to->getRoute();
        //cout <<this->getPoligonID()<< " ID , this->_eval_points " << this->_eval_points<< " route_poligon.back() " << route_poligon.back()  << "  end_2"<<endl;
        //Compara el poligono que esta pisando la punta de tracker con el 
        //this->_eval_points = esto es el poligono el cual se esta evaluando , guardado en el config
        //route_poligon.back() = rect tipo poligono , bounding box , ultimo 
    
        /////
        auto bboxPoints = route_poligon.back(); // This is std::vector<cv::Point2f>
        cv::Point2f bboxCenter((bboxPoints[0].x + bboxPoints[2].x) / 2.0f, (bboxPoints[0].y + bboxPoints[2].y) / 2.0f);
        double isInside = cv::pointPolygonTest(this->_eval_points, bboxCenter, false);
        bool status_1 = isInside >= 0;
        if(status_1==1){
            to->setPolygon(this->getPoligonID());
            
        }
        if (route_area_size <  6 || !to->isAwake()) {
            continue;
        }
            //float status_1 = isInside >= 0;  // true if bboxCenter is inside or on edge; false if outside
          int route_size = route_area.size();
          //cout <<route_size <<endl;
        auto bboxPoints_first = route_poligon[route_size-6];
        cv::Point2f bboxCenter_first((bboxPoints_first[0].x + bboxPoints_first[2].x) / 2.0f, (bboxPoints_first[0].y + bboxPoints_first[2].y) / 2.0f);
        double isInside_0 = cv::pointPolygonTest(this->_eval_points, bboxCenter_first, false);
        // int width = 640; // Example width, adjust as needed
        //int height = 480; // Example height, adjust as needed
        // cv::Mat darkImage = cv::Mat::zeros(cv::Size(width, height), CV_8UC3);
        //auto drawPolygon = [&darkImage](const std::vector<cv::Point2f>& points, const cv::Scalar& color) {
        //std::vector<cv::Point> intPoints;
        //for (const auto& p : points) {
        //    intPoints.push_back(cv::Point(static_cast<int>(p.x), static_cast<int>(p.y)));
        //}
        //const cv::Point* pts = (const cv::Point*) cv::Mat(intPoints).data;
        //int npts = cv::Mat(intPoints).rows;

        //cv::polylines(darkImage, &pts, &npts, 1, true, color, 2, cv::LINE_AA);
    //};

    // Draw evaluation polygon (_eval_points)
    //drawPolygon(this->_eval_points, cv::Scalar(0, 255, 0)); // Green

    //for (size_t i = 0; i < number_tracker; ++i) {
    ///    TrackingObject* to = this->_trackers[i];
    //    if (!to->isActive()) { continue; }

        // Draw the last bounding box (route_poligon.back())
    //    if (!to->getBboxPoligon().empty()) {
    //        drawPolygon(to->getBboxPoligon().back(), cv::Scalar(0, 0, 255)); // Red
    //    }

        // Draw the first bounding box (route_poligon.front())
    //    if (!to->getBboxPoligon().empty()) {
    //        drawPolygon(route_poligon[route_size-2], cv::Scalar(255, 0, 0)); // Blue
    //    }
    //}

    // Display the result
    //cv::imshow("Evaluation Visualization", darkImage);
    //cv::waitKey(1); 
      bool status_0 = isInside_0 >= 0;
            if (status_0 == 1 && status_1 == 1) { continue; }

        std::string type_event;
        bool flag = false;

        if ((status_0 == 0 && status_1 == 1)) {
            flag = true;
            type_event = "IN";

            
        } 

        if ((status_0 == 1 && status_1 == 0)) {
            flag = true;
            type_event = "OUT";
        } 

        if (flag) {
            vector<Point> route_center = to->getRoute();
            size_t route_center_size = route_center.size();

            Point p1 = route_center[route_center_size-1];
            Point p0 = route_center[route_center_size-2];     

            Point p0_ext;
            p0_ext.x = 1.5*p0.x - 0.5*p1.x;
            p0_ext.y = 1.5*p0.y - 0.5*p1.y;

            Point p1_ext;
            p1_ext.x = 1.5*p1.x - 0.5*p0.x;
            p1_ext.y = 1.5*p1.y - 0.5*p0.y;

            int lado = -1;
            cv::Point2f inter_point;     

            size_t n_points = this->_eval_points.size();
            for (size_t j = 0; j < n_points; ++j) {
                cv::Point2f p2 = _eval_points[j];
                cv::Point2f p3 = _eval_points[(j + 1) % n_points];
                cv::Point2f pout;
                bool k = calculateIntersection(p0_ext,p1_ext,p2,p3,pout);
                if(k){
                    lado = j;
                    Json::Value event_poligon;
                    event_poligon["type_event"] = type_event;
                    event_poligon["side"] = lado;
                    //cout <<this->getPoligonID()<< " ID , this->_eval_points   , LADO :  " << lado <<"  TYPE  : " <<type_event<< "   " <<to->getId() << " Tracker ID _ Time" <<to_string(getTimeMilis())<<endl;
                    if(this->getPoligonID()=="17" && lado==2 && type_event=="OUT"){
                    cout <<this->getPoligonID()<< " ID , this->_eval_points   , LADO :  " << lado <<"  TYPE  : " <<type_event<< "   " <<to->getId() << " Tracker ID _ Time" <<to_string(getTimeMilis())<<endl;
                    }
                    if(this->getPoligonID()=="17" && lado==0&& type_event=="IN"){
                    cout <<this->getPoligonID()<< " ID , this->_eval_points   , LADO :  " << lado <<"  TYPE  : " <<type_event<< "   " <<to->getId() << " Tracker ID _ Time" <<to_string(getTimeMilis())<<endl;
                    }
                    Json::Value temp_point;
                    // se normaliza el punto de intercepcion con respecto a las dimensiones de la imagen
                    temp_point["x"] = pout.x / static_cast<float>(this->getSize().width);
                    temp_point["y"] = pout.y / static_cast<float>(this->getSize().height);
                    //
                    event_poligon["interception_point"] = temp_point;
                    event_poligon["poligon_id"] = this->getPoligonID();
                    to->setPolygonEvent(event_poligon);
                    break;
                }
            }
    }

}
}