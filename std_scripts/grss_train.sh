# Multitask: depth and semantics

export PYTHONWARNINGS="ignore"
export CUDA_VISIBLE_DEVICES=$1

# dataset="nyu_795_1225x919_gaussian_bmvc"
# dataset="nyu_795_1225x919"
dataset='2018IEEE_Contest'
dataset_name="dfc"
max_d=30 #1.0
dfc_preprocessing=0
outputs_nc='1 21'
which_raster='dsm_demtli'

# Training
begin=1
step=15
end=121
g_lr=0.0002
batch_size=6
imageSize=320

# Network
net_architecture="D3net_multitask"
d_block_type='basic'

# Visualization
port=$2
display_id=$3
display_freq=1000
val_freq=2000000 # 5000

weightDecay=0
init_method="normal"
scale=1.0
alpha=0.5 # weight of regression
tasks='depth semantics'
# mtl_method='mgda_approx'
# mtl_method='mgda'
mtl_method=$4 #'eweights'
model="multitask" # multitask, regression, semantics
tasks='depth semantics' # semantics, depth

# when realizing tests
# batch_size=2

name=$mtl_method"_"$model"_"$dataset_name"_"$netG"_DA_TTFFT_320x320__mb"$batch_size"_prep"$dfc_preprocessing'_noskips'

python ./main_raster.py --name $name --dataroot ./datasets/$dataset/Phase2 --dataset_name $dataset_name --mtl_method $mtl_method --batchSize $batch_size --imageSize $imageSize --nEpochs $end --save_checkpoint_freq $step --model $model --display_id $display_id --display_freq $display_freq --port $port --net_architecture $net_architecture --val_freq $val_freq --val_split val --use_resize --use_dropout --pretrained --data_augmentation t t f f t --max_distance $max_d --scale_to_mm $scale --train --tasks $tasks --alpha $alpha --outputs_nc $outputs_nc --dfc_preprocessing $dfc_preprocessing --which_raster $which_raster #--use_skips

# name=$mtl_method"_"$model"_"$dataset_name"_"$netG"_DA_TTFFT_320x320__mb"$batch_size"_prep"$dfc_preprocessing'_skips'

# python ./main_raster.py --name $name --dataroot ./datasets/ISPRS/$dataset --dataset_name $dataset_name --mtl_method $mtl_method --batchSize $batch_size --imageSize $imageSize --nEpochs $end --save_checkpoint_freq $step --model $model --display_id $display_id --port $port --net_architecture $net_architecture --val_freq $val_freq --val_split val --use_resize --use_dropout --pretrained --data_augmentation t t f f t --max_distance $max_d --scale_to_mm $scale --train --tasks $tasks --alpha $alpha --outputs_nc $outputs_nc --dfc_preprocessing $dfc_preprocessing --use_skips
