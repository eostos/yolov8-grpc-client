#!/bin/bash\n/app/build/object-detection-triton-cpp-client $@\ntail -f /dev/null
#!/bin/bash
# Your script commands here
./build/computer-vision-triton-cpp-client     --source=/app/output.mp4     --task_type=detection     --model_type=yolov8    --model=yolov8-trt-traffic-2    --labelsFile=./coco.names   --input_sizes="3 640 640"  --protocol=grpc     --serverAddress=0.0.0.0     --port=8001
