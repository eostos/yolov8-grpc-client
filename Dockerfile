# Use an official Nvidia Triton Inference Server image as the base image
FROM nvcr.io/nvidia/tritonserver:23.08-py3-sdk

# Install any additional dependencies if needed
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libcurl4-openssl-dev \
    rapidjson-dev \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV TritonClientBuild_DIR /workspace/install

# Copy your C++ source code and CMakeLists.txt into the container

COPY . /app

# Set the working directory
WORKDIR /app

# Build your C++ application
RUN rm -rf build && mkdir build && cd build && cmake -DCMAKE_BUILD_TYPE=Release .. && cmake --build .

# Create a script to keep the container alive
RUN echo "#!/bin/bash\n\
/app/build/object-detection-triton-cpp-client \$@\n\
tail -f /dev/null" > /app/start.sh \
    && chmod +x /app/start.sh

# Set the entry point for the container
ENTRYPOINT ["/app/start.sh"]

