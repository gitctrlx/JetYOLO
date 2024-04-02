./build/xtrt/build \
    "./xtrt/weights/yolov5s_ccpd.onnx" \
    "./xtrt/engine/yolov5s_ccpd.plan" \
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

./tools/taoconverter/nx/tao-converter \
    ./xtrt/weights/lprnet_epoch-05.etlt -p image_input,1x3x48x96,4x3x48x96,16x3x48x96 -e ./xtrt/engine/lprnet.plan -k nvidia_tlt 