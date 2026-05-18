"""
Cell Dataset V2 - Patch Training with Stratified Sampling

This dataset implements patch-based training for cell staining transfer with balanced sampling:
- Random crop 256x256 patches from original resolution images
- Stratified sampling strategy to handle all patch types:
  * 5% background patches (>95% background) - teaches model to keep background clean
  * 5% dense cell patches (<10% background) - teaches model to handle cell-dense regions
  * 90% normal mixed patches - regular training data
- Maintains 1:1 aspect ratio to avoid distorting cell morphology
- Preserves high-frequency texture information

Key improvements from naive sampling:
- Prevents "fake cells" in background regions (model sees background during training)
- Prevents "washed-out colors" in dense cell regions (model sees dense cells during training)
- More robust to distribution shifts between training and inference

Key differences from cell_dataset (V1):
- V1: resize_and_crop to 286x286, then crop to 256x256
- V2: random crop 256x256 patches directly from original resolution
"""
import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image, ImageDraw
import random
import numpy as np
import sc_unsb.utils.util as util


class CellDatasetV2(BaseDataset):
    """
    Dataset class for cell staining style transfer with patch training.

    Special features:
    1. Random crop 256x256 patches from original resolution images
    2. Background filtering: reject patches with background ratio > threshold
    3. B-domain repeat sampling: iterate B twice while iterating A once
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')

        if opt.phase == "test" and not os.path.exists(self.dir_A) \
           and os.path.exists(os.path.join(opt.dataroot, "valA")):
            self.dir_A = os.path.join(opt.dataroot, "valA")
            self.dir_B = os.path.join(opt.dataroot, "valB")

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)

        # Background filtering parameters
        self.background_threshold = getattr(opt, 'background_threshold', 0.95)
        self.max_attempts = getattr(opt, 'max_patch_attempts', 10)
        self.patch_size = opt.crop_size if hasattr(opt, 'crop_size') else 256

        # Edge detection parameters (v4.1 - improved background detection)
        self.enable_edge_check = getattr(opt, 'enable_edge_check', False)  # Disabled by default for training
        self.edge_threshold = getattr(opt, 'edge_threshold', 25)  # Edge magnitude threshold

        # Background sampling: sample 5% background patches to help model learn to keep background clean
        self.bg_sample_prob = getattr(opt, 'bg_sample_prob', 0.05)

        # Dense cell sampling: sample 5-10% dense cell patches (patches with <10% background)
        # This helps model learn to handle patches that are completely filled with cells
        self.cell_sample_prob = getattr(opt, 'cell_sample_prob', 0.05)
        self.dense_cell_threshold = getattr(opt, 'dense_cell_threshold', 0.10)  # <10% background = dense cell

        # Print dataset info
        print(f"CellDatasetV2: A_size={self.A_size}, B_size={self.B_size}")
        print(f"  Patch size: {self.patch_size}x{self.patch_size}")
        print(f"  Background threshold: {self.background_threshold}")
        print(f"  Max patch attempts: {self.max_attempts}")
        print(f"  Edge check: {'ENABLED' if self.enable_edge_check else 'DISABLED'} (threshold={self.edge_threshold})")
        print(f"  Background sample prob: {self.bg_sample_prob} (5% = recommended)")
        print(f"  Dense cell sample prob: {self.cell_sample_prob} (5-10% = recommended)")
        print(f"  Dense cell threshold: {self.dense_cell_threshold} (<10% background = dense cell)")

    def is_valid_patch(self, img_array):
        """
        Check if a patch has valid content (not too much background).

        Parameters:
            img_array (np.ndarray) - patch as numpy array, shape (H, W, 3)

        Returns:
            bool - True if patch has sufficient content
        """
        # Convert to grayscale if needed
        if len(img_array.shape) == 3:
            gray = img_array.mean(axis=2)
        else:
            gray = img_array

        # Consider background as pixels close to white (255) or black (0)
        # For cell images, background is typically light/white
        background_mask = (gray > 240) | (gray < 15)
        background_ratio = background_mask.sum() / background_mask.size

        # Check background ratio first (quick rejection)
        if background_ratio >= self.background_threshold:
            return False

        # Optional: Check for edges/texture (v4.1)
        # This helps distinguish between:
        # - Pure background (low edges) -> invalid
        # - Background with noise/artifacts (high edges) -> valid for training
        if self.enable_edge_check:
            try:
                from scipy import ndimage
                # Use Sobel filter to detect edges
                sobel_h = ndimage.sobel(gray, axis=0)
                sobel_v = ndimage.sobel(gray, axis=1)
                edge_magnitude = np.sqrt(sobel_h**2 + sobel_v**2).mean()
            except ImportError:
                # Fallback if scipy is not available - use simple gradient
                gradient_h = np.abs(np.diff(gray, axis=0)).mean()
                gradient_v = np.abs(np.diff(gray, axis=1)).mean()
                edge_magnitude = gradient_h + gradient_v

            # If edge check is enabled, patches with BOTH high background ratio AND low edge activity
            # are considered invalid (pure background)
            # But patches with high background ratio BUT high edge activity (noise/artifacts)
            # are considered valid (should be trained on)
            if background_ratio > 0.70 and edge_magnitude < self.edge_threshold:
                return False

        # Valid if passed all checks
        return True

    def is_dense_cell_patch(self, img_array):
        """
        Check if a patch is densely filled with cells (very little background).

        Parameters:
            img_array (np.ndarray) - patch as numpy array, shape (H, W, 3)

        Returns:
            bool - True if patch has <10% background (i.e., >90% cells)
        """
        # Convert to grayscale if needed
        if len(img_array.shape) == 3:
            gray = img_array.mean(axis=2)
        else:
            gray = img_array

        # Calculate background ratio
        background_mask = (gray > 240) | (gray < 15)
        background_ratio = background_mask.sum() / background_mask.size

        # Dense cell patch if background ratio is very low
        return background_ratio < self.dense_cell_threshold

    def get_random_crop(self, img):
        """
        Get a random crop from image, with stratified sampling strategy.

        Sampling strategy:
        - bg_sample_prob (5%): sample pure background patches (>95% background)
        - cell_sample_prob (5%): sample dense cell patches (<10% background)
        - remaining (90%): sample normal mixed patches

        This balanced sampling helps the model learn to handle all three scenarios properly.

        Parameters:
            img (PIL.Image) - input image

        Returns:
            PIL.Image - cropped patch
        """
        w, h = img.size
        crop_size = self.patch_size

        # If image is smaller than crop size, resize it first
        if w < crop_size or h < crop_size:
            scale = max(crop_size / w, crop_size / h)
            new_w, new_h = int(w * scale), int(h * scale)
            img = img.resize((new_w, new_h), Image.BICUBIC)
            w, h = new_w, new_h

        # Decide which sampling mode to use
        rand = random.random()

        if rand < self.bg_sample_prob:
            # Mode 1: Sample background patch (5% probability)
            sample_mode = 'background'
        elif rand < (self.bg_sample_prob + self.cell_sample_prob):
            # Mode 2: Sample dense cell patch (5% probability)
            sample_mode = 'dense_cell'
        else:
            # Mode 3: Sample normal mixed patch (90% probability)
            sample_mode = 'normal'

        # Try to find a patch matching the target mode
        for attempt in range(self.max_attempts):
            left = random.randint(0, max(0, w - crop_size))
            top = random.randint(0, max(0, h - crop_size))
            right = left + crop_size
            bottom = top + crop_size

            crop = img.crop((left, top, right, bottom))
            crop_array = np.array(crop)

            # Check patch type
            is_bg = not self.is_valid_patch(crop_array)
            is_dense = self.is_dense_cell_patch(crop_array)

            if sample_mode == 'background' and is_bg:
                # Found a background patch as intended
                return crop
            elif sample_mode == 'dense_cell' and is_dense:
                # Found a dense cell patch as intended
                return crop
            elif sample_mode == 'normal' and not is_bg and not is_dense:
                # Found a normal mixed patch as intended
                return crop

        # If no suitable patch found after max_attempts, return the last attempt
        return crop

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int) -- a random integer for data indexing

        Returns:
            dictionary - contains A, B, A_paths, B_paths
        """
        A_path = self.A_paths[index % self.A_size]

        if self.opt.serial_batches:
            # For serial batches, iterate B twice per A iteration
            index_B = (index // 2) % self.B_size
        else:
            # For random batches, use deterministic sampling
            seed = index + self.current_epoch * 10000
            rng = random.Random(seed)
            index_B = rng.randint(0, self.B_size - 1)

        B_path = self.B_paths[index_B]

        # Load images
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        # Apply random crop with background filtering
        A_crop = self.get_random_crop(A_img)
        B_crop = self.get_random_crop(B_img)

        # Convert to tensor with normalization
        # Note: we don't use get_transform here since we do custom cropping
        A = self.crop_to_tensor(A_crop)
        B = self.crop_to_tensor(B_crop)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def crop_to_tensor(self, img):
        """Convert PIL image to tensor with normalization.

        Parameters:
            img (PIL.Image) - input image

        Returns:
            torch.Tensor - normalized tensor
        """
        import torch
        from torchvision import transforms

        # Convert to tensor and normalize to [-1, 1]
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        return transform(img)

    def __len__(self):
        """Return the total number of images in the dataset.

        For training, use A_size. For each A image, B is sampled twice.
        """
        if self.opt.phase == 'train':
            return self.A_size
        else:
            return max(self.A_size, self.B_size)

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add dataset-specific options.

        Parameters:
            parser -- argument parser
            is_train (bool) -- whether training phase

        Returns:
            modified parser
        """
        parser.add_argument('--background_threshold', type=float, default=0.95,
                            help='Maximum ratio of background pixels in a patch (0-1)')
        parser.add_argument('--max_patch_attempts', type=int, default=10,
                            help='Maximum attempts to find a valid patch')

        # v4.1: Edge detection parameters for improved background detection
        parser.add_argument('--enable_edge_check', action='store_true',
                            help='Enable edge detection in background check (disabled by default for training)')
        parser.add_argument('--edge_threshold', type=float, default=25,
                            help='Edge magnitude threshold for background detection (default: 25)')

        parser.add_argument('--bg_sample_prob', type=float, default=0.05,
                            help='Probability of sampling background patches (default: 0.05 = 5%%)')
        parser.add_argument('--cell_sample_prob', type=float, default=0.05,
                            help='Probability of sampling dense cell patches (default: 0.05 = 5%%)')
        parser.add_argument('--dense_cell_threshold', type=float, default=0.10,
                            help='Background ratio threshold for dense cell patches (default: 0.10 = <10%% background)')

        parser.set_defaults(no_flip=True)  # Disable flip for cell images
        parser.set_defaults(preprocess='none')  # We handle preprocessing ourselves

        return parser
