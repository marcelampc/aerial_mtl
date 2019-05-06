# simplified main

from options.extra_args_dfc import MTL_Raster_Options
from dataloader.data_loader import CreateDataLoader
from models.getmodels import create_model
from ipdb import set_trace as st
from dataloader.dataset_raster import DatasetFromFolder as Dataset

# Load options
opt = MTL_Raster_Options().parse()
# train model
if opt.train or opt.resume:
    if 'eweights' in opt.mtl_method or 'alexnorm' in opt.mtl_method:
        from models.mtl_raster import RasterMTL as Model
    elif 'mgda' in opt.mtl_method:
        from models.mtl_raster_mgda import RasterMTL_MGDA as Model
    elif 'gradnorm' in opt.mtl_method:
        from models.mtl_raster_gradnorm import MTLGradNorm as Model
    elif 'gp' in opt.mtl_method:
        from models.mtl_raster_gp import RasterMTL_GP as Model
    model = Model()
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    data_loader, val_loader = CreateDataLoader(opt, Dataset)
    model.train(data_loader, val_loader=val_loader)
elif opt.test:
    if 'dfc' in opt.dataset_name:
        from models.test_model_raster import TestModel
    else:
        from models.test_model_raster_isprs import TestModel
    model = TestModel()
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    model.test_raster()
