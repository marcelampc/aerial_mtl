# For Make3D datasets
# Original image dimensions:
# RGB: 1704x2272  Depth: 305x55
# Dimensions to the network:
# 256x320 both. for tests, we will change. Keep same network.

export CUDA_VISIBLE_DEVICES=$1

dataset='Vaihingen'
# astroboy
port=$2
display_id=$3
begin=1
step=20
end=301
dataset_name="isprs"
# dataset=$dataset_name"_val"
method="reg"
method_reg=L1 # implement
# model="raster_regression"
model="raster_regression"
netG="DenseUNet"
# netG="D3net_multitask_residual" cant
#netG="resUnet50"
d_lr=0.0002
g_lr=0.0002
d_block_type='basic'
weightDecay=0
init_method="normal"
batch_size=6
scale=1.0 
max_d=1.0
val_freq=30000
dfc_preprocessing=3 # locally sub min value
mask_thres=9000
# pretrained_path="/data2/mcarvalh/softs/cgan/models/our_pretrained_models/denseUnet121_pt_gan__regL1__320x256__mb8_noDropoutnyu_230k_u16"

# When using astroboy, NEVER use the flag --use_cudnn_benchmark

name="grss_"$dataset_name"_"$netG"_"$d_block_type'_'$model"_DA_TTFFT_"$method_reg"__320x320__mb"$batch_size"_lr_"$g_lr"_drop_pt_"$dataset"_prep"$dfc_preprocessing'_'$which_raster'_mask_0_9000_rot'

python ./main_raster.py --dataroot ./datasets/ISPRS/$dataset --reg_type $method_reg --init_method $init_method --name $name --batchSize $batch_size --imageSize 320 320 --nEpochs $end --save_checkpoint_freq $step --save_samples --model $model --use_reg --use_lsgan --use_cgan --display_id $display_id --port $port --display_freq 10 --which_model_netG $netG --use_$method --g_lr $g_lr --weightDecay $weightDecay --d_block_type $d_block_type --use_skips --train --val_freq $val_freq --dataset_name $dataset_name --val_split val --use_resize --use_dropout --pretrained --data_augmentation t t f f t --max_distance $max_d --scale_to_mm $scale --dfc_preprocessing $dfc_preprocessing --mask_thres $mask_thres #--validate
