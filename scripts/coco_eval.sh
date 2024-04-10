./build/xtrt/tools/eval/yolo_eval \
    "./xtrt/engine/yolo.plan" \
    "./xtrt/data/coco/filelist.txt" \
    "./xtrt/data/coco/val2017/" \
    "results.json" \
    0 \
    true \
    2 \
    1 3 640 640

python3 ./xtrt/tools/eval/eval_coco.py