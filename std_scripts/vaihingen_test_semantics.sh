# For Make3D datasets
# Original image dimensions:
# RGB: 1704x2272  Depth: 305x55
# Dimensions to the network:
# 256x320 both. for tests, we will change. Keep same network.

# dataset="nyu_795_1225x919_gaussian_bmvc"
# dataset="nyu_795_1225x919"
export CUDA_VISIBLE_DEVICES=$2
dataset='Vaihingen'
dataset_name="isprs"
model="raster_semantics"
scale=1.0 #65.535
test_split=test
epoch=$3
dfc_preprocessing=3

name=$1

patch_size=1024
test_stride=256 #1024 #1024 #$patch_size
reconstruction_method=gaussian #concatenation

python ./main_raster.py --imageSize $patch_size $patch_size --dataroot ./datasets/ISPRS/$dataset --name $name --save_samples --test --test_split $test_split --port 0 --display_id 0 --batchSize 1 --epoch ${epoch} --model $model --use_skips --dataset_name $dataset_name --input_nc 3 --scale_to_mm $scale --dfc_preprocessing $dfc_preprocessing --test_stride $test_stride --reconstruction_method $reconstruction_method --use_semantics --n_classes 7 --save_semantics #--normalize #--use_padding 

# offset=0.0
# python ../scripts/error_measurement/measure_raster.py --path_gt ./datasets/$dataset/Phase2 --path_pred ./results/dfc/$name/$epoch/output/ --name $name --which_raster $which_raster --phase test #--normalize #--use_semantics # --offset $
