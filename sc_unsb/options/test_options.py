from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        # Dropout and Batchnorm has different behavioir during training and test.
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        parser.add_argument('--num_test', type=int, default=50, help='how many test images to run')
        # Inference mode for V1 experiment (resize, resize_back, patch)
        parser.add_argument('--v1_mode', type=str, default='resize',
                            help='inference mode for V1: resize (256x256 output), resize_back (original resolution), patch (patch-based)')
        # Patch-based inference parameters
        parser.add_argument('--patch_size', type=int, default=256, help='patch size for patch-based inference')
        parser.add_argument('--overlap', type=int, default=64, help='overlap between patches')
        # Background detection threshold (lower = more aggressive filtering to avoid pseudo-cells)
        parser.add_argument('--bg_threshold', type=float, default=0.70,
                            help='patches with >X%% background skip inference (default: 0.70 for 70%% background)')
        # Disable background detection for pure inference
        parser.add_argument('--disable_bg_detection', action='store_true',
                            help='disable background detection and run pure inference on all patches')

        # Dense Normalization (SC-UNSB) parameters
        parser.add_argument('--use_dn', action='store_true',
                            help='use Dense Normalization for spatially-continuous inference (SC-UNSB)')

        # To avoid cropping, the load_size should be the same as crop_size
        parser.set_defaults(load_size=parser.get_default('crop_size'))
        self.isTrain = False
        return parser
