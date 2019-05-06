# Based on cycle gan
# later transform in class like in marti


def create_model(opt):
    print(opt.model)
    if opt.test or opt.visualize:
        if opt.use_semantics:
            from .test_semantics_model import TestSemanticsModel
            model = TestSemanticsModel()
        else:
            from .test_model import TestModel
            model = TestModel()
    else:
        if 'regression' in opt.model:
            from .regression import RegressionModel
            model = RegressionModel()
        elif opt.model == 'gan':
            from .gan_model import GANModel
            model = GANModel()
        elif opt.model == 'wgan':
            from .wgan_model import WGANModel
            model = WGANModel()
        elif opt.model == 'diw':
            from .depth_wild import DIWModel
            model = DIWModel()
        elif opt.model == 'two_streams':
            from .regression_2streams import RegressionModelTwoStreams
            model = RegressionModelTwoStreams()
        elif opt.model == 'depth_and_semantics':
            from .regression_semantics import RegressionSemanticsModel
            model = RegressionSemanticsModel()
        elif opt.model == 'regression_multiscale':
            from .regression_multiscale import RegressionMultiscaleModel
            model = RegressionMultiscaleModel()
        else:
            raise ValueError("Model [%s] not recognized." % opt.model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
# class CreateModel(object):
