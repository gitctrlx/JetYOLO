./build/xtrt/build \
    "./xtrt/weights/yolov5s_trt8.onnx" \
    "./xtrt/engine/yolo_trt8.plan" \
    "fp16" \
    3 \
    1 1 1 \
    3 3 3 \
    640 640 640 \
    640 640 640 \
    550 \
    "./xtrt/data/coco/val2017" \
    "./xtrt/data/coco/filelist.txt" \
    "./xtrt/engine/int8Cache/int8.cache" \
    true \
    false \
    "./xtrt/engine/timingCache/timing.cache"
