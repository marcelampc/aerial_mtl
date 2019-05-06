# For Make3D datasets
# Original image dimensions:
# RGB: 1704x2272  Depth: 305x55
# Dimensions to the network:
# 256x320 both. for tests, we will change. Keep same network.

# dataset="nyu_795_1225x919_gaussian_bmvc"
# dataset="nyu_795_1225x919"
dataset='208IEEE_Contest'
# astroboy
port=$1
display_id=$2
begin=1
step=10000000
end=4000
dataset_name="dfc"
# dataset=$dataset_name"_val"
method="reg"
method_reg=L1 # implement
model="regression"
netG="D3net_simple"
#netG="resUnet50"
d_lr=0.0002
g_lr=0.0002
d_block_type='basic'
weightDecay=0
init_method="normal"
batch_size=8
scale=1000.0 #65.535
max_d=10.0
val_freq=500000000
# pretrained_path="/data2/mcarvalh/softs/cgan/models/our_pretrained_models/denseUnet121_pt_gan__regL1__320x256__mb8_noDropoutnyu_230k_u16"

# When using astroboy, NEVER use the flag --use_cudnn_benchmark

name="grssdfc_"$netG$d_block_type"_DA_TTTT_"$method_reg"__320x256__mb"$batch_size"_drop_pt_"$dataset

python ./main.py --dataroot ./datasets/$dataset/Phase2 --reg_type $method_reg --init_method $init_method --name $name --batchSize $batch_size --imageSize 320 256 --nEpochs $end --save_checkpoint_freq $step --save_samples --model $model --use_reg --use_lsgan --use_cgan --display_id $display_id --val_freq $val_freq --validate --val_split test --port $port --display_freq 10 --which_model_netG $netG --use_$method --g_lr $g_lr --weightDecay $weightDecay --d_block_type $d_block_type --use_skips --train --dataset_name $dataset_name --use_resize --use_dropout --pretrained --data_augmentation t t t t --max_distance $max_d --scale_to_mm $scale 
