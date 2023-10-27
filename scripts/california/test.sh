export CUDA_VISIBLE_DEVICES=0

model=resnet101
path_train=California-All-Aug-RN101-100-Epochs
data_path=/home/ad084/ad084_media/california/img_divided/

python3 eval_knn.py --arch $model --pretrained_weights /home/ad084/ad084_media/dino/${path_train}/checkpoint.pth \
        --checkpoint_key teacher --data_path $data_path > /home/ad084/ad084_media/dino/${path_train}/log_eval_knn.txt

python3 eval_linear.py --arch $model --epochs 20 \
        --pretrained_weights /home/ad084/ad084_media/dino/${path_train}/checkpoint.pth \
        --checkpoint_key teacher --data_path $data_path > /home/ad084/ad084_media/dino/${path_train}/log_eval_linear.txt