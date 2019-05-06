# % load_ext autoreload
# % autoreload 2

def save_rasters(root='./', filename='merged'):
    import rasterio
    from rasterio.merge import merge
    from rasterio.plot import show
    from os.path import join
    import argparse
    import glob
    files = glob.glob(join(root, '*.tif'))
    mosaic_rasters = [rasterio.open(file) for file in files]

    mosaic, out_transform = merge(mosaic_rasters)

    meta = (rasterio.open(files[0])).meta

    meta.update({"driver": "GTiff",
                "height": mosaic.shape[1],
                "width":  mosaic.shape[2],
                "transform": out_transform})
    # show(mosaic)
    with rasterio.open(join(root, '{}.tif'.format(filename)), "w", **meta) as dest:
        dest.write(mosaic)

def save_merged_rasters(root, fileroot=None):
    import rasterio
    from rasterio.merge import merge
    from rasterio.plot import show
    from os.path import join
    import argparse
    import glob
    
    root = '{}*.tif'.format(fileroot)
    filename = 'merged_{}.tif'.format(fileroot)

    files = glob.glob(join(root))
    mosaic_rasters = [rasterio.open(file) for file in files]

    mosaic, out_transform = merge(mosaic_rasters)

    meta = (rasterio.open(files[0])).meta

    meta.update({"driver": "GTiff",
                "height": mosaic.shape[1],
                "width":  mosaic.shape[2],
                "transform": out_transform})

    with rasterio.open(filename, "w", **meta) as dest:
        dest.write(mosaic)

    # filename = '{}/{}/{}_merged.png'.format(root, datatype, datatype) 
    # self.save_raster_png(mosaic, filename)

    if 'output' in filename or 'target' in filename:
        save_height_colormap(filename, mosaic)

def save_height_colormap(filename, data, cmap='jet', max_v=40):
    import matplotlib.pyplot as plt
    plt.switch_backend('agg')
    dpi = 80
    data = data[0,:,:]
    height, width = data.shape 
    figsize = width / float(dpi), height / float(dpi)
    # change string
    # if 'output' in filename:
    #     filename = filename.replace('merged_output', 'cmap_merged_output')
    # else:
    filename = filename.replace('merged_', 'cmap_merged_')
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    cax = ax.imshow(data, vmax=max_v, vmin=0, aspect='auto', interpolation='spline16',
    cmap=cmap)
    # cax = ax.imshow(data, vmax=0.1, vmin=0, aspect='auto', interpolation='spline16',
    # cmap=cmap)
    ax.set(xlim=[0, width], ylim=[height, 0], aspect=1)

    fig.savefig(filename, dpi=dpi)
    del fig, data

def color_rasters(cmap='jet', max_v=40):
    import rasterio
    # filename='./merged_target.tif'
    filename = '/data2/mcarvalh/results/results_cgan/dfc/eweights__DA_TTFFT_320x320__mb6_prep0_NOval_mask_0_9000_noundef_rotation90_real/0120/bayesian/merged_m_error.tif'
    data = rasterio.open(filename).read()
    save_height_colormap(filename, data, cmap, max_v=max_v)
