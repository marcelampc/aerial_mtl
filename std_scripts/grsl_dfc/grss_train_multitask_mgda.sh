# Multitask: depth and semantics

export CUDA_VISIBLE_DEVICES=$1

# dataset="nyu_795_1225x919_gaussian_bmvc"
# dataset="nyu_795_1225x919"
dataset='2018IEEE_Contest'
dataset_name="dfc"
max_d=30 #1.0
dfc_preprocessing=$4
outputs_nc='1 21'
which_raster='dsm_demtli'

dataset='Vaihingen'
dataset_name='vaihingen'
max_d=1.0
dfc_preprocessing=3
outputs_nc='1 7'

# Training and network
begin=1
step=15
end=61
d_lr=0.0002
g_lr=0.0002
batch_size=4
imageSize=320
# Network
net_architecture="D3net_multitask"
d_block_type='basic'

# Visualization
port=$2
display_id=$3
val_freq=2000000 # 5000

weightDecay=0
init_method="normal"
scale=1.0 #1.0 #65.535
mask_thres=9000 # if we are normalizing
alpha=0.5 # weight of regression
mtl_method=$4 #'eweights'
model="multitask" # multitask, regression, semantics
tasks='depth semantics' # semantics, depth

name=$mtl_method"_"$model"_"$dataset_name"_"$netG"_DA_TTFFT_320x320__mb"$batch_size"_prep"$dfc_preprocessing'_NOval_mask_0_9000_noundef_rotation90_real'

# python ./main_raster.py --name $name --dataroot ./datasets/$dataset/Phase2 --dataset_name $dataset_name --mtl_method $mtl_method --batchSize $batch_size --imageSize $imageSize --nEpochs $end --save_checkpoint_freq $step --model $model --display_id $display_id --port $port --net_architecture $net_architecture --val_freq $val_freq --val_split val --use_resize --use_dropout --pretrained --data_augmentation t t f f t --max_distance $max_d --scale_to_mm $scale --train --tasks $tasks --alpha $alpha --outputs_nc $outputs_nc --which_raster $which_raster --dfc_preprocessing $dfc_preprocessing # --use_skips

python ./main_raster.py --name $name --dataroot ./datasets/ISPRS/$dataset --dataset_name $dataset_name --mtl_method $mtl_method --batchSize $batch_size --imageSize $imageSize --nEpochs $end --save_checkpoint_freq $step --model $model --display_id $display_id --port $port --net_architecture $net_architecture --val_freq $val_freq --val_split val --use_resize --use_dropout --pretrained --data_augmentation t t f f t --max_distance $max_d --scale_to_mm $scale --train --tasks $tasks --alpha $alpha --outputs_nc $outputs_nc --which_raster $which_raster --dfc_preprocessing $dfc_preprocessing # --use_skips