from .extra_args_mtl import MTL_Options

class MTL_Raster_Options(MTL_Options):
    def initialize(self):
        MTL_Options.initialize(self)
        self.parser.add_argument('--dfc_preprocessing', type=int, default=30)
        self.parser.add_argument('--which_raster', default='dsm', help='optios are: dsm, dsm-demb, dsm-dem3msr, dsm-demtli')
        self.parser.add_argument('--test_stride', type=int, nargs='+', default=[128])
        self.parser.add_argument('--reconstruction_method', default='concatenation', help='concatenation, gaussian')
        self.parser.add_argument('--save_semantics', action='store_true')
        self.parser.add_argument('--save_target', action='store_true')
        self.parser.add_argument('--mean_rotation', type=float, default=0.0)
        self.parser.add_argument('--max_rotation', type=float, default=5.0)