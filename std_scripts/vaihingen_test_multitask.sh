
export CUDA_VISIBLE_DEVICES=$2
dataset='Vaihingen'
dataset_name="isprs"
model="raster_multitask"
scale=1.0 #65.535
test_split=test
epoch=$3
dfc_preprocessing=3

name=$1

patch_size=1024
test_stride=1024 #1024 #1024 #$patch_size
reconstruction_method=gaussian #concatenation
export PYTHONWARNINGS="ignore"

python ./main_raster.py --imageSize $patch_size $patch_size --dataroot ./datasets/ISPRS/$dataset --name $name --save_samples --test --test_split $test_split --port 0 --display_id 0 --batchSize 1 --epoch ${epoch} --model $model --dataset_name $dataset_name --input_nc 3 --scale_to_mm $scale --dfc_preprocessing $dfc_preprocessing --test_stride $test_stride --reconstruction_method $reconstruction_method --n_classes 7 --save_semantics --use_skips #--normalize #--use_padding --use_skips

echo='SEMANTICS METRICS'
python ../scripts/error_measurement/measure_raster_semantics.py --path_gt ./datasets/ISPRS_BENCHMARK_DATASETS/$dataset --path_pred ./results/$dataset_name/$name/$epoch/output/ --name $name --phase test --n_classes 7 

echo='REGRESSION METRICS'
python ../scripts/error_measurement/measure_raster.py --path_gt ./datasets/ISPRS/$dataset --path_pred ./results/$dataset_name/$name/$epoch/output/ --name $name --phase test #--normalize #--use_semantics # --offset $


# offset=0.0
# python ../scripts/error_measurement/measure_raster.py --path_gt ./datasets/$dataset/Phase2 --path_pred ./results/dfc/$name/$epoch/output/ --name $name --which_raster $which_raster --phase test #--normalize #--use_semantics # --offset $
