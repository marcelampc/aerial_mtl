# Multitask: depth and semantics

export CUDA_VISIBLE_DEVICES=$1

# dataset="nyu_795_1225x919_gaussian_bmvc"
# dataset="nyu_795_1225x919"
dataset='Vaihingen'
# astroboy
port=$2
display_id=$3
begin=1
step=20
end=101
dataset_name="isprs"
method="reg"
method_reg=L1 # implement
model="raster_multitask"
netG="D3net_multitask"
# netG="D3net_multitask_residual"
# netG="D3net_multitask_3scale"
d_lr=0.0002
g_lr=0.0002
d_block_type='basic'
weightDecay=0
init_method="normal"
batch_size=6
scale=1.0 #1.0 #65.535
max_d=1.0 #1.0
val_freq=2000 # 5000
dfc_preprocessing=3
mask_thres=9000 # if we are normalizing
alpha=0.5 # weight of regression
n_classes=7

alpha=0.5 # weight of regression
tasks='depth semantics'
outputs_nc='1 21'
# mtl_method='mgda_approx'
mtl_method='mgda'
model="multitask"
imageSize=320

name="grss_"$dataset_name"_"$netG$d_block_type"_DA_TTFFT_320x320__mb"$batch_size"_lr_"$g_lr"_drop_pt_"$dataset"_prep"$dfc_preprocessing'_'$which_raster'_NOval_multitask_alpha_'$alpha'_mask_rot_nDSM'

python ./main_raster.py --dataroot ./datasets/ISPRS/$dataset --init_method $init_method --name $name --batchSize $batch_size --imageSize 320 320 --nEpochs $end --save_checkpoint_freq $step --save_samples --model $model --use_reg --use_lsgan --use_cgan --display_id $display_id --port $port --display_freq 100 --which_model_netG $netG --use_$method --g_lr $g_lr --weightDecay $weightDecay --d_block_type $d_block_type --use_skips --val_freq $val_freq --dataset_name $dataset_name --val_split val --use_resize --use_dropout --pretrained --data_augmentation t t f f t --max_distance $max_d --scale_to_mm $scale --dfc_preprocessing $dfc_preprocessing --mask_thres $mask_thres --use_semantics --alpha $alpha --n_classes $n_classes --train #--resume #--validate 

