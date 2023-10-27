export CUDA_VISIBLE_DEVICES=0
export MASTER_PORT=29500

python main_dino.py --arch resnet101 --local_crops_number 0 --global_crops_scale 0.14 1 --aug_gauss_blur False --aug_color True --aug_flip False --epochs 100 --batch_size_per_gpu 12 --optimizer sgd --warmup_teacher_temp_epochs 15 --teacher_temp 0.06 --lr 0.01 --weight_decay 1e-4 --weight_decay_end 1e-6 --data_path /home/ad084/ad084_media/california/img_divided/train --output_dir /home/ad084/ad084_media/dino/California-Only-Color-RN101-100-Epochs