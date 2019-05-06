# Multitask: depth and semantics

export CUDA_VISIBLE_DEVICES=$1

# dataset="nyu_795_1225x919_gaussian_bmvc"
# dataset="nyu_795_1225x919"
dataset='2018IEEE_Contest'
# astroboy
port=$2
display_id=$3
begin=1
step=20
end=300
dataset_name="dfc"
method="reg"
method_reg=L1 # implement
model="raster_multitask"
#netG="D3net_multitask"
net_architecture="D3net_multitask"
# netG="D3net_multitask_3scale"
d_lr=0.0002
g_lr=0.0002
d_block_type='basic'
weightDecay=0
init_method="normal"
# batch_size=1
batch_size=1
scale=1.0 #1.0 #65.535
max_d=30 #1.0
val_freq=2000000 # 5000
dfc_preprocessing=$4
mask_thres=9000 # if we are normalizing
which_raster='dsm_demtli'
alpha=0.5 # weight of regression
tasks='depth semantics'
outputs_nc='1 21'
mtl_method=eweights
model="multitask"
imageSize=320

name=$mtl_method"_$dataset_name_"$netG"_DA_TTFFT_320x320__mb"$batch_size"_prep"$dfc_preprocessing'_NOval_mask_0_9000_noundef_rotation90_real'

python ./main_raster.py --name $name --dataroot ./datasets/$dataset/Phase2 --dataset_name $dataset_name --mtl_method $mtl_method --batchSize $batch_size --imageSize $imageSize --nEpochs $end --save_checkpoint_freq $step --model $model --display_id $display_id --port $port --net_architecture $net_architecture --val_freq $val_freq --val_split val --use_resize --use_dropout --pretrained --data_augmentation t t f f t --max_distance $max_d --scale_to_mm $scale --train --tasks $tasks --alpha $alpha --outputs_nc $outputs_nc --which_raster $which_raster --dfc_preprocessing $dfc_preprocessing #--use_skips