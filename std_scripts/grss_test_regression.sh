# For Make3D datasets
# Original image dimensions:
# RGB: 1704x2272  Depth: 305x55
# Dimensions to the network:
# 256x320 both. for tests, we will change. Keep same network.

# dataset="nyu_795_1225x919_gaussian_bmvc"
# dataset="nyu_795_1225x919"
export CUDA_VISIBLE_DEVICES=$3
dataset='2018IEEE_Contest'
begin=1
step=200
end=2000
dataset_name="dfc"
# dataset=$dataset_name"_val"
method="reg"
model="raster_regression"
d_block_type='basic'
scale=1.0 #65.535
max_d=30.0
dfc_preprocessing=1
test_split=test
epoch=$4
dfc_preprocessing=$2
# pretrained_path="/data2/mcarvalh/softs/cgan/models/our_pretrained_models/denseUnet121_pt_gan__regL1__320x256__mb8_noDropoutnyu_230k_u16"

# When using astroboy, NEVER use the flag --use_cudnn_benchmark

name=$1
which_raster='dsm_demtli'

patch_size=1024
test_stride=1024 #$patch_size
reconstruction_method=gaussian

python ./main_raster.py --imageSize $patch_size $patch_size --dataroot ./datasets/$dataset/Phase2 --name $name --save_samples --test --test_split $test_split --port 0 --display_id 0 --batchSize 1 --epoch ${epoch} --model $model --d_block_type $d_block_type --use_skips --dataset_name $dataset_name --input_nc 3 --scale_to_mm $scale --dfc_preprocessing $dfc_preprocessing --test_stride $test_stride --reconstruction_method $reconstruction_method --which_raster $which_raster #--use_padding 

# offset=0.0
python ../scripts/error_measurement/measure_raster.py --path_gt ./datasets/$dataset/Phase2 --path_pred ./results/dfc/$name/$epoch/output/ --name $name --which_raster $which_raster --phase test # --offset $
