python ./mmyolo/projects/easydeploy/tools/export_onnx.py \
	./mmyolo/configs/yolov5/yolov5_s-v61_syncbn_8xb16-300e_coco.py \
	yolov5s.pth \
	--work-dir work_dir \
    --img-size 640 640 \
    --batch 1 \
    --device cpu \
    --simplify \
	--opset 11 \
	--backend tensorrt8 \ 
	--pre-topk 1000 \
	--keep-topk 100 \
	--iou-threshold 0.65 \
	--score-threshold 0.25

python ./mmyolo/projects/easydeploy/tools/export_onnx.py \
	./mmyolo/configs/yolov5/yolov5_s-v61_syncbn_8xb16-300e_coco.py \
	yolov5s.pth \
	--work-dir work_dir \
    --img-size 640 640 \
    --batch 1 \
    --device cpu \
    --simplify \
	--opset 11 \
	--model-only