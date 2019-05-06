# Bayesian Network
# Epistemic Uncertainty

export CUDA_VISIBLE_DEVICES=$1
dataset='2018IEEE_Contest'
dataset_name="dfc"
# dataset=$dataset_name"_val"
model="raster_multitask"
scale=1.0 #65.535
test_split=test
epoch=$3

name=$2
which_raster='dsm_demtli'

patch_size=512
test_stride=128 #1024 #1024 #$patch_size
reconstruction_method=gaussian #concatenation

python ./bayesian_test_raster.py --imageSize $patch_size $patch_size --dataroot ./datasets/$dataset/Phase2 --name $name --save_samples --test --test_split $test_split --port 0 --display_id 0 --batchSize 1 --epoch ${epoch} --model $model --use_skips --dataset_name $dataset_name --input_nc 3 --scale_to_mm $scale --test_stride $test_stride --reconstruction_method $reconstruction_method --use_semantics --n_classes 21 --which_raster $which_raster
