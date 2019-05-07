# For Make3D datasets
# Original image dimensions:
# RGB: 1704x2272  Depth: 305x55
# Dimensions to the network:
# 256x320 both. for tests, we will change. Keep same network.

# dataset="nyu_795_1225x919_gaussian_bmvc"
# dataset="nyu_795_1225x919"
export CUDA_VISIBLE_DEVICES=$2
dataset='2018IEEE_Contest'
dataset_name="dfc"
scale=1.0 #65.535
test_split=test
epoch=$3
name=$1
which_raster='dsm_demtli'

patch_size=1024
test_stride=256 #1024 #1024 #$patch_size
reconstruction_method=gaussian #concatenation

export PYTHONWARNINGS="ignore"
# export CUDA_LAUNCH_BLOCKING=1

python ./main_raster.py --imageSize $patch_size $patch_size --dataroot ./datasets/$dataset/Phase2 --name $name --test --test_split $test_split --port 0 --display_id 0 --batchSize 1 --epoch ${epoch} --use_skips --dataset_name $dataset_name --input_nc 3 --scale_to_mm $scale --test_stride $test_stride --reconstruction_method $reconstruction_method --save_semantics --n_classes 21 --save_samples #--normalize #--use_padding 

python ../scripts/error_measurement/measure_raster.py --path_gt ./datasets/$dataset/Phase2 --path_pred ./results/dfc/$name/$epoch/output/ --name $name --which_raster $which_raster --phase test


# def save_height_colormap(filename, data, cmap='jet'):
#     import matplotlib.pyplot as plt
#     plt.switch_backend('agg')
#     dpi = 80
#     data = data[0,:,:]
#     height, width = data.shape 
#     figsize = width / float(dpi), height / float(dpi)
#     # change string
#     if 'output' in filename:
#         filename = filename.replace('merged_output', 'cmap_merged_output')
#     else:
#         filename = filename.replace('merged_target', 'cmap_merged_target')
#     fig = plt.figure(figsize=figsize)
#     ax = fig.add_axes([0, 0, 1, 1])
#     ax.axis('off')
#     cax = ax.imshow(data, vmax=30, vmin=0, aspect='auto', interpolation='spline16', cmap=cmap)
#     ax.set(xlim=[0, width], ylim=[height, 0], aspect=1)

#     fig.savefig(filename, dpi=dpi)
#     del fig, data

# import rasterio
# filename='./merged_output.tif'
# data = rasterio.open(filename).read()
# save_height_colormap(filename, data)
