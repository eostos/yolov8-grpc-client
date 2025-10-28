// main_video_with_imshow.cpp
#include <cuda_runtime_api.h>
#include "grpc_client.h"
#include <unistd.h>
#include <iostream>
#include <string>
#include <random>
#include <sstream>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <chrono>
#include <thread>
#include "shm_utils.h"

namespace tc = triton::client;

#define FAIL_IF_ERR(X, MSG)                                        \
  {                                                                \
    tc::Error err = (X);                                           \
    if (!err.IsOk()) {                                             \
      std::cerr << "error: " << (MSG) << ": " << err << std::endl; \
      exit(1);                                                     \
    }                                                              \
  }

#define FAIL_IF_CUDA_ERR(FUNC)                                     \
  {                                                                \
    const cudaError_t result = FUNC;                               \
    if (result != cudaSuccess) {                                   \
      std::cerr << "CUDA exception (line " << __LINE__             \
                << "): " << cudaGetErrorName(result) << " ("       \
                << cudaGetErrorString(result) << ")" << std::endl; \
      exit(1);                                                     \
    }                                                              \
  }

void CreateCUDAIPCHandle(cudaIpcMemHandle_t* cuda_handle, void* input_d_ptr, int device_id = 0) {
    FAIL_IF_CUDA_ERR(cudaSetDevice(device_id));
    FAIL_IF_CUDA_ERR(cudaIpcGetMemHandle(cuda_handle, input_d_ptr));
}

std::string generateUniqueName(const std::string& prefix) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<> dis(10000, 99999);
    
    std::stringstream ss;
    ss << prefix << "_" << dis(gen);
    return ss.str();
}

// Función para limpiar memoria compartida existente
void cleanupExistingSharedMemory(std::unique_ptr<tc::InferenceServerGrpcClient>& client, 
                                const std::string& base_name) {
    // Intentar desregistrar cualquier memoria compartida existente
    try {
        client->UnregisterCudaSharedMemory(base_name + "_input");
        std::cout << "Limpiada memoria compartida: " << base_name + "_input" << std::endl;
    } catch (...) {
        // Ignorar errores si no existe
    }
    
    try {
        client->UnregisterCudaSharedMemory(base_name + "_output");
        std::cout << "Limpiada memoria compartida: " << base_name + "_output" << std::endl;
    } catch (...) {
        // Ignorar errores si no existe
    }
}

// Función para preprocesar el frame
cv::Mat preprocessFrame(const cv::Mat& frame, const cv::Size& target_size) {
    cv::Mat resized, float_frame;
    
    // Redimensionar al tamaño esperado por el modelo (640x640)
    cv::resize(frame, resized, target_size);
    
    // Convertir de BGR a RGB
    cv::Mat rgb_frame;
    cv::cvtColor(resized, rgb_frame, cv::COLOR_BGR2RGB);
    
    // Normalizar a [0, 1] y convertir a float32
    rgb_frame.convertTo(float_frame, CV_32FC3, 1.0/255.0);
    
    return float_frame;
}

// Función para procesar detecciones YOLO y dibujar bounding boxes
void processYOLODetections(const float* output_data, const std::vector<int64_t>& shape, 
                          cv::Mat& display_frame, int frame_count) {
    // shape: [1, 9, 8400]
    
    int num_detections = shape[2]; // 8400
    int num_attributes = shape[1]; // 9
    
    int valid_detections = 0;
    float confidence_threshold = 0.5;
    
    // Colores para diferentes clases
    std::vector<cv::Scalar> colors = {
        cv::Scalar(0, 255, 0),    // Verde
        cv::Scalar(255, 0, 0),    // Azul
        cv::Scalar(0, 0, 255),    // Rojo
        cv::Scalar(255, 255, 0),  // Cian
        cv::Scalar(255, 0, 255),  // Magenta
        cv::Scalar(0, 255, 255),  // Amarillo
        cv::Scalar(128, 0, 128),  // Púrpura
        cv::Scalar(255, 165, 0),  // Naranja
        cv::Scalar(128, 128, 0),  // Oliva
        cv::Scalar(0, 128, 128)   // Verde azulado
    };
    
    // Nombres de clases (ajusta según tu modelo)
    std::vector<std::string> class_names = {
        "person", "bicycle", "car", "motorcycle", "airplane",
        "bus", "train", "truck", "boat", "traffic light"
    };
    
    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> class_ids;
    
    for (int i = 0; i < num_detections; ++i) {
        const float* detection = output_data + i * num_attributes;
        float confidence = detection[4];
        
        if (confidence > confidence_threshold) {
            // Encontrar clase con mayor probabilidad
            int class_id = 0;
            float max_class_prob = 0.0;
            for (int j = 5; j < num_attributes; ++j) {
                if (detection[j] > max_class_prob) {
                    max_class_prob = detection[j];
                    class_id = j - 5;
                }
            }
            
            // Solo considerar si la probabilidad de clase es alta
            if (max_class_prob > 0.5) {
                float x_center = detection[0];
                float y_center = detection[1];
                float width = detection[2];
                float height = detection[3];
                
                // Convertir de coordenadas normalizadas YOLO (centro) a píxeles (esquina)
                int x1 = static_cast<int>((x_center - width/2) * display_frame.cols);
                int y1 = static_cast<int>((y_center - height/2) * display_frame.rows);
                int x2 = static_cast<int>((x_center + width/2) * display_frame.cols);
                int y2 = static_cast<int>((y_center + height/2) * display_frame.rows);
                
                // Asegurar que las coordenadas están dentro de la imagen
                x1 = std::max(0, std::min(x1, display_frame.cols - 1));
                y1 = std::max(0, std::min(y1, display_frame.rows - 1));
                x2 = std::max(0, std::min(x2, display_frame.cols - 1));
                y2 = std::max(0, std::min(y2, display_frame.rows - 1));
                
                // Solo agregar si el bounding box es válido
                if (x2 > x1 && y2 > y1) {
                    boxes.emplace_back(x1, y1, x2 - x1, y2 - y1);
                    confidences.push_back(confidence * max_class_prob);
                    class_ids.push_back(class_id);
                    valid_detections++;
                }
            }
        }
    }
    
    // Aplicar Non-Maximum Suppression para eliminar detecciones duplicadas
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, confidence_threshold, 0.4, indices);
    
    // Dibujar detecciones en el frame
    for (int idx : indices) {
        cv::Rect box = boxes[idx];
        float conf = confidences[idx];
        int class_id = class_ids[idx];
        
        // Seleccionar color basado en la clase
        cv::Scalar color = colors[class_id % colors.size()];
        
        // Dibujar rectángulo
        cv::rectangle(display_frame, box, color, 2);
        
        // Crear etiqueta
        std::string label = class_names[class_id % class_names.size()] + 
                           " " + cv::format("%.2f", conf);
        
        // Dibujar fondo para la etiqueta
        int baseline;
        cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        cv::rectangle(display_frame, 
                     cv::Point(box.x, box.y - label_size.height - baseline),
                     cv::Point(box.x + label_size.width, box.y),
                     color, cv::FILLED);
        
        // Dibujar texto
        cv::putText(display_frame, label, 
                   cv::Point(box.x, box.y - baseline), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    }
    
    // Agregar información del frame
    std::string info_text = "Frame: " + std::to_string(frame_count) + 
                           " | Detecciones: " + std::to_string(indices.size());
    cv::putText(display_frame, info_text, 
               cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
    
    if (indices.size() > 0) {
        std::cout << "Frame " << frame_count << ": " << indices.size() << " detecciones válidas" << std::endl;
    }
}

void processVideo(const std::string& video_path, std::string url, bool verbose, tc::Headers http_headers) {
    // Abrir el video
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "Error: No se puede abrir el video: " << video_path << std::endl;
        return;
    }
    
    // Obtener propiedades del video
    double fps = cap.get(cv::CAP_PROP_FPS);
    int width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    int total_frames = cap.get(cv::CAP_PROP_FRAME_COUNT);
    
    std::cout << "Procesando video: " << video_path << std::endl;
    std::cout << "Resolución: " << width << "x" << height << std::endl;
    std::cout << "FPS: " << fps << std::endl;
    std::cout << "Total frames: " << total_frames << std::endl;
    
    // Crear ventana para mostrar el video
    cv::namedWindow("YOLO Object Detection - Triton Server", cv::WINDOW_NORMAL);
    cv::resizeWindow("YOLO Object Detection - Triton Server", 800, 600);
    
    // Configuración del cliente
    std::unique_ptr<tc::InferenceServerGrpcClient> client;
    FAIL_IF_ERR(
        tc::InferenceServerGrpcClient::Create(&client, url, verbose),
        "unable to create grpc client");

    // Generar nombres únicos para esta sesión
    std::string session_id = generateUniqueName("video_session");
    std::string input_shm_name = session_id + "_input";
    std::string output_shm_name = session_id + "_output";
    
    std::cout << "Usando nombres de memoria compartida: " << input_shm_name << ", " << output_shm_name << std::endl;

    // Limpiar memoria compartida existente
    cleanupExistingSharedMemory(client, "video_session");

    std::string model_name = "water-jugs";
    std::string model_version = "";
    std::string input_tensor_name = "images";
    std::string output_tensor_name = "output0";
    
    int64_t batch_size = 1;
    std::vector<int64_t> shape{batch_size, 3, 640, 640};
    size_t input_byte_size = batch_size * 3 * 640 * 640 * sizeof(float);
    size_t output_byte_size = batch_size * 9 * 8400 * sizeof(float);
    
    // Configurar memoria GPU
    float* input_d_ptr = nullptr;
    float* output0_d_ptr = nullptr;
    
    FAIL_IF_CUDA_ERR(cudaMalloc((void**)&input_d_ptr, input_byte_size));
    FAIL_IF_CUDA_ERR(cudaMalloc((void**)&output0_d_ptr, output_byte_size));
    
    // Registrar memoria compartida
    cudaIpcMemHandle_t input_cuda_handle, output_cuda_handle;
    CreateCUDAIPCHandle(&input_cuda_handle, input_d_ptr);
    CreateCUDAIPCHandle(&output_cuda_handle, output0_d_ptr);
    
    FAIL_IF_ERR(
        client->RegisterCudaSharedMemory(input_shm_name, input_cuda_handle, 0, input_byte_size),
        "failed to register input shared memory");
    
    FAIL_IF_ERR(
        client->RegisterCudaSharedMemory(output_shm_name, output_cuda_handle, 0, output_byte_size),
        "failed to register output shared memory");
    
    std::cout << "Memoria compartida registrada exitosamente" << std::endl;
    
    // Procesar frames del video
    cv::Mat frame, processed_frame, display_frame;
    int frame_count = 0;
    const int max_frames = 200; // Aumentar límite para ver más del video
    
    double total_inference_time = 0.0;
    bool paused = false;
    
    std::cout << "\nControles:" << std::endl;
    std::cout << " - Presiona ESPACIO para pausar/reanudar" << std::endl;
    std::cout << " - Presiona 'q' o ESC para salir" << std::endl;
    std::cout << " - Presiona 's' para guardar frame actual" << std::endl;
    
    while (cap.read(frame) && frame_count < max_frames) {
        if (frame.empty()) {
            std::cout << "Frame vacío detectado, terminando..." << std::endl;
            break;
        }
        
        frame_count++;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Crear copia para mostrar (no modificar el frame original)
        display_frame = frame.clone();
        
        // Preprocesar frame para inferencia
        processed_frame = preprocessFrame(frame, cv::Size(640, 640));
        
        // Copiar datos a GPU
        FAIL_IF_CUDA_ERR(cudaMemcpy(
            input_d_ptr, processed_frame.data, input_byte_size, cudaMemcpyHostToDevice));
        
        // Crear input para este frame
        tc::InferInput* input0;
        FAIL_IF_ERR(
            tc::InferInput::Create(&input0, input_tensor_name, shape, "FP32"),
            "unable to get input tensor");
        std::shared_ptr<tc::InferInput> input0_ptr;
        input0_ptr.reset(input0);
        
        FAIL_IF_ERR(
            input0_ptr->SetSharedMemory(input_shm_name, input_byte_size, 0),
            "unable to set shared memory for input");
        
        // Configurar output
        tc::InferRequestedOutput* output0;
        FAIL_IF_ERR(
            tc::InferRequestedOutput::Create(&output0, output_tensor_name),
            "unable to get output tensor");
        std::shared_ptr<tc::InferRequestedOutput> output0_ptr;
        output0_ptr.reset(output0);
        
        FAIL_IF_ERR(
            output0_ptr->SetSharedMemory(output_shm_name, output_byte_size, 0),
            "unable to set shared memory for output");
        
        // Ejecutar inferencia
        tc::InferOptions options(model_name);
        options.model_version_ = model_version;
        
        std::vector<tc::InferInput*> inputs = {input0_ptr.get()};
        std::vector<const tc::InferRequestedOutput*> outputs = {output0_ptr.get()};
        
        tc::InferResult* results;
        auto infer_err = client->Infer(&results, options, inputs, outputs, http_headers);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        double inference_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();
        total_inference_time += inference_time;
        
        if (!infer_err.IsOk()) {
            std::cerr << "Frame " << frame_count << ": Error en inferencia - " << infer_err << std::endl;
            
            // Mostrar frame sin detecciones
            cv::putText(display_frame, "ERROR EN INFERENCIA", 
                       cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
        } else {
            std::shared_ptr<tc::InferResult> results_ptr;
            results_ptr.reset(results);
            
            // Procesar resultados y dibujar detecciones
            const float* output_data;
            size_t output_byte_size_received;
            if (results_ptr->RawData(output_tensor_name, (const uint8_t**)&output_data, &output_byte_size_received).IsOk()) {
                std::vector<int64_t> output_shape;
                if (results_ptr->Shape(output_tensor_name, &output_shape).IsOk()) {
                    processYOLODetections(output_data, output_shape, display_frame, frame_count);
                }
            }
            
            // Agregar información de tiempo
            std::string time_text = "Tiempo: " + cv::format("%.1f", inference_time) + " ms";
            cv::putText(display_frame, time_text, 
                       cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
            
            std::cout << "Frame " << frame_count << " procesado en " << inference_time << " ms" << std::endl;
        }
        
        // Mostrar el frame con detecciones
        cv::imshow("YOLO Object Detection - Triton Server", display_frame);
        
        // Manejar controles de teclado
        int key = cv::waitKey(1) & 0xFF;
        
        if (key == 'q' || key == 27) { // 'q' o ESC
            std::cout << "Salida solicitada por usuario" << std::endl;
            break;
        } else if (key == ' ') { // Espacio para pausar/reanudar
            paused = !paused;
            std::cout << (paused ? "Video pausado" : "Video reanudado") << std::endl;
        } else if (key == 's') { // Guardar frame
            std::string filename = "frame_" + std::to_string(frame_count) + ".jpg";
            cv::imwrite(filename, display_frame);
            std::cout << "Frame guardado como: " << filename << std::endl;
        }
        
        // Si está pausado, esperar hasta que el usuario presione espacio nuevamente
        while (paused) {
            key = cv::waitKey(100) & 0xFF;
            if (key == ' ' || key == 'q' || key == 27) {
                if (key == ' ') {
                    paused = false;
                    std::cout << "Video reanudado" << std::endl;
                } else {
                    std::cout << "Salida solicitada por usuario" << std::endl;
                    break;
                }
            }
        }
        
        // Pequeña pausa para mantener timing del video
        if (fps > 0) {
            int delay = std::max(1, static_cast<int>(1000.0 / fps) - static_cast<int>(inference_time));
            std::this_thread::sleep_for(std::chrono::milliseconds(delay));
        }
    }
    
    // Estadísticas finales
    if (frame_count > 0) {
        std::cout << "\n=== ESTADÍSTICAS FINALES ===" << std::endl;
        std::cout << "Frames procesados: " << frame_count << std::endl;
        std::cout << "Tiempo total de inferencia: " << total_inference_time << " ms" << std::endl;
        std::cout << "Tiempo promedio por frame: " << (total_inference_time / frame_count) << " ms" << std::endl;
        std::cout << "FPS aproximado: " << (1000.0 / (total_inference_time / frame_count)) << std::endl;
    }
    
    // Limpieza
    std::cout << "Limpiando memoria compartida..." << std::endl;
    client->UnregisterCudaSharedMemory(input_shm_name);
    client->UnregisterCudaSharedMemory(output_shm_name);
    
    if (input_d_ptr) cudaFree(input_d_ptr);
    if (output0_d_ptr) cudaFree(output0_d_ptr);
    
    cap.release();
    cv::destroyAllWindows();
    std::cout << "Procesamiento de video completado." << std::endl;
}

int main(int argc, char** argv) {
    bool verbose = false;
    std::string url("localhost:8001");
    std::string video_path("test_video.mp4");
    tc::Headers http_headers;

    // Parse arguments
    int opt;
    while ((opt = getopt(argc, argv, "vu:H:i:")) != -1) {
        switch (opt) {
            case 'v':
                verbose = true;
                break;
            case 'u':
                url = optarg;
                break;
            case 'H': {
                std::string arg = optarg;
                std::string header = arg.substr(0, arg.find(":"));
                http_headers[header] = arg.substr(header.size() + 1);
                break;
            }
            case 'i':
                video_path = optarg;
                break;
            default:
                std::cerr << "Uso: " << argv[0] << " [-v] [-u URL] [-i video_path] [-H header]" << std::endl;
                return 1;
        }
    }

    // Verificar que el archivo de video existe
    std::ifstream test_file(video_path);
    if (!test_file.good()) {
        std::cerr << "Error: El archivo de video no existe: " << video_path << std::endl;
        std::cerr << "Usa: -i /ruta/completa/al/video.mp4" << std::endl;
        return 1;
    }
    test_file.close();

    std::cout << "=== TRITON VIDEO PROCESSING CLIENT ===" << std::endl;
    std::cout << "Video: " << video_path << std::endl;
    std::cout << "Model: water-jugs" << std::endl;
    std::cout << "Server: " << url << std::endl;
    std::cout << "======================================" << std::endl;

    // Procesar el video
    processVideo(video_path, url, verbose, http_headers);
    
    return 0;
}