#pragma once
#include <redox.hpp>

#include <opencv2/opencv.hpp>
#include <iostream>
#include "Utils.hpp"
#include "EventQueue.hpp"


namespace redisCallbacks {

// Declara si lo tienes global

inline auto lam_gotmsg = [](const std::string& topic, const std::string& msg) {
    	
    FrMs msgi;

    // Parsear JSON
    Json::CharReaderBuilder builder;
    Json::Value j;
    std::string errs;
    std::istringstream s(msg);
    bool ok = Json::parseFromStream(builder, s, &j, &errs);
    if (!ok) {
        std::cerr << "Error parsing JSON: " << errs << std::endl;
        return;
    }

    // Asignar campos básicos
    msgi.host_uuid    = j.get("host_uuid", "none").asString();
    msgi.timestamp    = j.get("timestamp", "0").asString();
    msgi.frame_id     = j.get("frame_id", "none").asString();
    msgi.fps          = j.get("fps", 0.0).asFloat();
    msgi.resolution_x = j.get("resolution_x", 0).asInt();
    msgi.resolution_y = j.get("resolution_y", 0).asInt();
    msgi.analytics_results = j["analytics_results"];
    msgi.analytic_type = j.get("analytic_type", "none").asString();
    msgi.event_type    = j.get("event_type", "none").asString();

    // Decodificar imagen base64, si existe
    if (j.isMember("frame_b64") && j["frame_b64"].isString()) {
        std::string encoded = j["frame_b64"].asString();
        std::string decoded = base64_decode(encoded); // implementa o usa tu función base64_decode_vec
        std::vector<uchar> data(decoded.begin(), decoded.end());
        msgi.mat_frame = cv::imdecode(data, cv::IMREAD_COLOR);
    } else {
        msgi.mat_frame = cv::Mat();
    }

    // Ahora puedes usar msgi como antes:
    // smartQueue.addNewItem(msgi);

    // (opcional) Imprimir debug:
    msgi.printEvent();
	////////////////////////////////
};

inline auto lam_subbed = [](const std::string& topic) {
    std::cout << ">>> Subscribed to " << topic << std::endl;
};

inline auto lam_unsubbed = [](const std::string& topic) {
    std::cout << ">>> Unsubscribed from " << topic << std::endl;
};

inline auto lam_err_callback = [](const std::string& topic, const int &errnum) {
    std::cerr << ">>> LAM_ERR_CALLBACK " << topic << " errnum: " << errnum << std::endl;
    std::cerr << "REDIS ERROR ACTIVATES THE redis ERROR CALLBACK" << std::endl;
    exit(-3);
};

inline auto lam_rdxconn = [](const int& state) {
    std::cout << ">>>>>>>>>>>>>>>>>>> RDX.CONNECT CALLBACK CONNECTED STATE: " << state << std::endl;
};

inline auto lam_subconn = [](const int& state) {
    std::cout << ">>>>>>>>>>>>>>>>>>> SUB.CONNECT CALLBACK CONNECTED STATE: " << state << std::endl;
};

} // namespace alice
