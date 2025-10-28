// main_final.cpp
#include <cuda_runtime_api.h>
#include "grpc_client.h"
#include <unistd.h>
#include <iostream>
#include <string>
#include <random>
#include <sstream>
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

std::string generateUniqueName(const std::string& prefix, int region, int iteration) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<> dis(10000, 99999);
    
    std::stringstream ss;
    ss << prefix << "_" << region << "_" << iteration << "_" << dis(gen);
    return ss.str();
}

void request(int region, std::string url, bool verbose, tc::Headers http_headers) {
    std::cout << "Thread " << region << ": Iniciando..." << std::endl;
    
    // Variables CUDA - usar float para FP32
    float* input_d_ptr = nullptr;
    float* output0_d_ptr = nullptr;
    float* input_data = nullptr;

    try {
        std::unique_ptr<tc::InferenceServerGrpcClient> client;
        FAIL_IF_ERR(
            tc::InferenceServerGrpcClient::Create(&client, url, verbose),
            "unable to create grpc client");

        // PARÁMETROS CORREGIDOS según los errores
        std::string model_name = "water-jugs";
        std::string model_version = "";
        std::string input_tensor_name = "images";
        std::string output_tensor_name = "output0";

        // SOLO 1 iteración para pruebas
        for (size_t i = 0; i < 1; i++) {
            // Nombres únicos para evitar conflictos
            std::string input_shm_name = generateUniqueName("input_data", region, i);
            std::string output_shm_name = generateUniqueName("output_data", region, i);
            
            std::cout << "Thread " << region << " usando: " << input_shm_name << std::endl;

            int64_t batch_size = 1;
            // FORMA CORREGIDA: [1, 3, 640, 640] en lugar de [1, 3, 640, 384]
            std::vector<int64_t> shape{batch_size, 3, 640, 640};
            
            // Tamaños corregidos para 640x640
            size_t input_byte_size = batch_size * 3 * 640 * 640 * sizeof(float);
            size_t output_byte_size = batch_size * 840000; // Tamaño estimado para output

            // Crear input - FP32 con forma corregida
            tc::InferInput* input0;
            FAIL_IF_ERR(
                tc::InferInput::Create(&input0, input_tensor_name, shape, "FP32"),
                "unable to get input tensor");
            std::shared_ptr<tc::InferInput> input0_ptr;
            input0_ptr.reset(input0);

            // Inicializar datos de entrada - FORMA CORREGIDA
            input_data = new float[batch_size * 3 * 640 * 640];
            for (size_t j = 0; j < batch_size * 3 * 640 * 640; ++j) {
                input_data[j] = 0.0f;
            }

            // Memoria GPU para input
            FAIL_IF_CUDA_ERR(cudaMalloc((void**)&input_d_ptr, input_byte_size));
            FAIL_IF_CUDA_ERR(cudaMemcpy(
                input_d_ptr, input_data, input_byte_size, cudaMemcpyHostToDevice));

            // Registrar memoria compartida para input
            cudaIpcMemHandle_t input_cuda_handle;
            CreateCUDAIPCHandle(&input_cuda_handle, input_d_ptr);

            FAIL_IF_ERR(
                client->RegisterCudaSharedMemory(
                    input_shm_name, input_cuda_handle, 0, input_byte_size),
                "failed to register input shared memory");

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

            // Memoria GPU para output
            FAIL_IF_CUDA_ERR(cudaMalloc((void**)&output0_d_ptr, output_byte_size));

            cudaIpcMemHandle_t output_cuda_handle;
            CreateCUDAIPCHandle(&output_cuda_handle, output0_d_ptr);

            FAIL_IF_ERR(
                client->RegisterCudaSharedMemory(
                    output_shm_name, output_cuda_handle, 0, output_byte_size),
                "failed to register output shared memory");

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
            
            std::shared_ptr<tc::InferResult> results_ptr;
            
            if (!infer_err.IsOk()) {
                std::cerr << "Thread " << region << ": Error en inferencia - " << infer_err << std::endl;
                
                // Si falla por nombre de output, probar alternativos
                if (std::string(infer_err.Message()).find("output") != std::string::npos) {
                    std::cout << "Probando con diferentes nombres de output..." << std::endl;
                    
                    std::vector<std::string> output_names = {"output0", "output", "outputs", "detections", "boxes"};
                    bool success = false;
                    
                    for (const auto& output_name : output_names) {
                        std::cout << "Probando output: " << output_name << std::endl;
                        
                        tc::InferRequestedOutput* output_test;
                        if (tc::InferRequestedOutput::Create(&output_test, output_name).IsOk()) {
                            std::shared_ptr<tc::InferRequestedOutput> output_test_ptr;
                            output_test_ptr.reset(output_test);
                            
                            std::vector<const tc::InferRequestedOutput*> outputs_test = {output_test_ptr.get()};
                            auto test_err = client->Infer(&results, options, inputs, outputs_test, http_headers);
                            
                            if (test_err.IsOk()) {
                                std::cout << "✓ Output correcto: " << output_name << std::endl;
                                results_ptr.reset(results);
                                success = true;
                                break;
                            }
                        }
                    }
                    
                    if (!success) {
                        std::cerr << "No se encontró el nombre correcto del output" << std::endl;
                    }
                }
            } else {
                results_ptr.reset(results);
                std::cout << "Thread " << region << ": INFERENCIA EXITOSA con modelo '" << model_name << "'" << std::endl;
                
                // Mostrar información básica de los resultados
                if (results_ptr) {
                    std::cout << "✓ Inferencia completada exitosamente" << std::endl;
                    
                    // Verificar información del output
                    std::string datatype;
                    if (results_ptr->Datatype(output_tensor_name, &datatype).IsOk()) {
                        std::cout << "Output data type: " << datatype << std::endl;
                    }
                    
                    std::vector<int64_t> output_shape;
                    if (results_ptr->Shape(output_tensor_name, &output_shape).IsOk()) {
                        std::cout << "Output shape: ";
                        for (const auto& dim : output_shape) {
                            std::cout << dim << " ";
                        }
                        std::cout << std::endl;
                    }
                    
                    // CORRECCIÓN: Id() necesita un parámetro
                    std::string request_id;
                    if (results_ptr->Id(&request_id).IsOk()) {
                        std::cout << "Request ID: " << request_id << std::endl;
                    }
                }
            }

            // Limpieza inmediata
            client->UnregisterCudaSharedMemory(input_shm_name);
            client->UnregisterCudaSharedMemory(output_shm_name);

            if (input_d_ptr) {
                cudaFree(input_d_ptr);
                input_d_ptr = nullptr;
            }
            if (output0_d_ptr) {
                cudaFree(output0_d_ptr);
                output0_d_ptr = nullptr;
            }
            if (input_data) {
                delete[] input_data;
                input_data = nullptr;
            }
        }

        std::cout << "Thread " << region << ": Completado" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Thread " << region << ": ERROR - " << e.what() << std::endl;
        
        // Limpieza de emergencia
        if (input_d_ptr) cudaFree(input_d_ptr);
        if (output0_d_ptr) cudaFree(output0_d_ptr);
        if (input_data) delete[] input_data;
    }
}

int main(int argc, char** argv) {
    bool verbose = false;
    std::string url("localhost:8001");
    tc::Headers http_headers;

    int opt;
    while ((opt = getopt(argc, argv, "vu:H:")) != -1) {
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
            default:
                break;
        }
    }

    std::cout << "=== CLIENTE TRITON - VERSIÓN FINAL ===" << std::endl;
    std::cout << "Modelo: water-jugs" << std::endl;
    std::cout << "Tipo de datos: FP32" << std::endl;
    std::cout << "Forma de entrada: [1, 3, 640, 640]" << std::endl;
    std::cout << "Conectando a: " << url << std::endl;
    std::cout << "=====================================" << std::endl;

    const int num_threads = 1;
    std::vector<std::thread> threads;
    
    for (int i = 0; i < num_threads; ++i) {
        threads.push_back(std::thread(request, i, url, verbose, http_headers));
    }

    for (auto& t : threads) {
        t.join();
    }
    
    std::cout << "=== PRUEBA COMPLETADA ===" << std::endl;
    return 0;
}