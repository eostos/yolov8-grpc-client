#!/bin/bash\n/app/build/object-detection-triton-cpp-client $@\ntail -f /dev/null
#!/bin/bash
# Your script commands here
#./build/computer-vision-triton-cpp-client     --source=rtsp://190.84.115.70:8554/14-c6543190-e7b2-4ff0-9c49-242dd323de3f/1     --task_type=detection     --model_type=yolov8    --model=yolov8-trt    --labelsFile=./coco.names   --input_sizes="3 640 640"  --protocol=grpc     --serverAddress=0.0.0.0     --port=8001
./build/computer-vision-triton-cpp-client /opt/alice-config/config_tracker.json 1   #  --source=/app/output.mp4    --task_type=detection     --model_type=yolov8    --model=yolov8-trt-traffic-2    --labelsFile=./coco.names   --input_sizes="3 640 640"  --protocol=grpc     --serverAddress=0.0.0.0     --port=8001
