1.Install required libs run
pip3 install -r requirements.txt

2. Download pretrained coco weights
https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5

3.To run training execute

python3 --model path/to/coco/pretrained/weight.h5 --dataset-dir path/to/dir/where/dataset/images --dataset-config path/to/json