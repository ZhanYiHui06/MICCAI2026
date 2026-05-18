"""
SC-UNSB (Spatially-Continuous Unpaired Neural Schrödinger Bridge) Inference Script

This script implements Dense Normalization (DN) for seamless patch-based inference,
eliminating tiling artifacts at patch boundaries.

Two-pass workflow:
1. Pass 1 (Collection Mode): Collect instance normalization statistics for all patches
2. Pass 2 (Inference Mode): Use bilinearly interpolated statistics for smooth transitions

PLUG-AND-PLAY MODE:
    Models trained with standard Instance Normalization can use DN inference
    without retraining! The script automatically replaces IN layers with DN layers.

Usage:
    # For models trained with DN (--normG dn):
    python test_sc_unsb.py --name cell_sc_unsb --dataroot datasets/cell \
        --dataset_mode cell_v2 --direction BtoA --model sb --mode sb \
        --normG dn --use_dn --phase test --epoch latest

    # For models trained with standard IN (plug-and-play):
    python test_sc_unsb.py --name cell_v2 --dataroot datasets/cell \
        --dataset_mode cell_v2 --direction BtoA --model sb --mode sb \
        --use_dn --phase test --epoch latest
"""
import os
import ntpath
from collections import OrderedDict
import torch
import numpy as np
from PIL import Image
from options.test_options import TestOptions
from sc_unsb.data import create_dataset
from sc_unsb.models import create_model
from sc_unsb.utils import html
import sc_unsb.utils.util as util
from tqdm import tqdm


class SCUNSBInferencer:
    """
    SC-UNSB Inferencer with Dense Normalization for seamless patch-based inference.

    This class implements the two-pass DN workflow:
    1. Collection pass: Collect IN statistics for all patches
    2. Inference pass: Use interpolated statistics for smooth transitions
    """

    def __init__(self, model, patch_size=256, overlap=64, dn_padding=1,
                 bg_threshold=0.70, disable_bg_detection=False):
        """
        Initialize the SC-UNSB inferencer.

        Parameters:
            model - trained SB model
            patch_size (int) - size of each patch (default: 256)
            overlap (int) - overlap between patches (default: 64)
            dn_padding (int) - padding for DN interpolation (default: 1)
            bg_threshold (float) - background ratio threshold (default: 0.70)
            disable_bg_detection (bool) - if True, process all patches (default: False)
        """
        self.model = model
        self.patch_size = patch_size
        self.overlap = overlap
        self.stride = patch_size - overlap
        self.dn_padding = dn_padding
        self.bg_threshold = bg_threshold
        self.disable_bg_detection = disable_bg_detection

    def is_background_patch(self, patch_tensor):
        """
        Detect if a patch is pure background.

        Parameters:
            patch_tensor (torch.Tensor) - patch tensor of shape (1, 3, H, W) in range [-1, 1]

        Returns:
            bool - True if patch is considered background
        """
        patch_array = ((patch_tensor + 1) * 127.5).cpu().numpy()
        patch_array = patch_array.squeeze()
        patch_array = patch_array.transpose(1, 2, 0).astype(np.uint8)

        if len(patch_array.shape) == 3:
            gray = patch_array.mean(axis=2)
        else:
            gray = patch_array

        background_mask = (gray > 240) | (gray < 15)
        background_ratio = background_mask.sum() / background_mask.size

        try:
            from scipy import ndimage
            sobel_h = ndimage.sobel(gray, axis=0)
            sobel_v = ndimage.sobel(gray, axis=1)
            edge_magnitude = np.sqrt(sobel_h**2 + sobel_v**2).mean()
        except ImportError:
            gradient_h = np.abs(np.diff(gray, axis=0)).mean()
            gradient_v = np.abs(np.diff(gray, axis=1)).mean()
            edge_magnitude = gradient_h + gradient_v

        is_background = background_ratio > self.bg_threshold and edge_magnitude < 25
        return is_background

    def get_patch_grid(self, img_size):
        """
        Calculate patch grid dimensions.

        Parameters:
            img_size (tuple) - (width, height) of the image

        Returns:
            tuple - (n_rows, n_cols, patches_info)
            patches_info is a list of (row, col, left, top, right, bottom)
        """
        w, h = img_size
        n_h = max(1, int(np.ceil((h - self.overlap) / self.stride)))
        n_w = max(1, int(np.ceil((w - self.overlap) / self.stride)))

        patches_info = []
        for i in range(n_h):
            for j in range(n_w):
                left = j * self.stride
                top = i * self.stride

                if left + self.patch_size > w:
                    left = max(0, w - self.patch_size)
                if top + self.patch_size > h:
                    top = max(0, h - self.patch_size)

                right = left + self.patch_size
                bottom = top + self.patch_size

                patches_info.append((i, j, left, top, right, bottom))

        return n_h, n_w, patches_info

    def create_blend_mask(self):
        """Create a blending mask for smooth patch reconstruction."""
        mask = np.ones((self.patch_size, self.patch_size), dtype=np.float32)

        if self.overlap > 0:
            for i in range(self.overlap):
                alpha = 0.5 * (1 - np.cos(i / self.overlap * np.pi))
                mask[:, i] *= alpha
                mask[:, -(i+1)] *= alpha

            for i in range(self.overlap):
                alpha = 0.5 * (1 - np.cos(i / self.overlap * np.pi))
                mask[i, :] *= alpha
                mask[-(i+1), :] *= alpha

        return mask

    def tensor_to_image(self, tensor):
        """Convert tensor to PIL Image."""
        tensor = tensor.squeeze(0).cpu()
        tensor = (tensor + 1) / 2.0
        tensor = tensor.clamp(0, 1)
        img_array = (tensor.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        return Image.fromarray(img_array)

    def image_to_tensor(self, img):
        """Convert PIL Image to tensor."""
        from torchvision import transforms

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        tensor = transform(img).unsqueeze(0)
        if torch.cuda.is_available():
            tensor = tensor.cuda()

        return tensor

    def reconstruct_image(self, patches_info, results, original_size):
        """
        Reconstruct full image from patches with blending.

        Parameters:
            patches_info - list of (row, col, left, top, right, bottom) tuples
            results - list of result tensors
            original_size (tuple) - (width, height) of original image

        Returns:
            PIL.Image - reconstructed image
        """
        w, h = original_size

        accumulator = np.zeros((h, w, 3), dtype=np.float32)
        weight_map = np.zeros((h, w), dtype=np.float32)

        blend_mask = self.create_blend_mask()

        for (row, col, left, top, right, bottom), result_tensor in zip(patches_info, results):
            result_img = self.tensor_to_image(result_tensor)
            patch_array = np.array(result_img, dtype=np.float32)

            pw = right - left
            ph = bottom - top

            mask_resized = Image.fromarray((blend_mask * 255).astype(np.uint8))
            mask_resized = mask_resized.resize((pw, ph), Image.BILINEAR)
            mask_array = np.array(mask_resized).astype(np.float32) / 255.0

            mask_array = np.stack([mask_array] * 3, axis=2)

            region = accumulator[top:bottom, left:right]
            accumulator[top:bottom, left:right] = region + patch_array * mask_array
            weight_map[top:bottom, left:right] += mask_array[:, :, 0]

        weight_map = np.stack([weight_map] * 3, axis=2)
        weight_map = np.maximum(weight_map, 1e-8)
        reconstructed = accumulator / weight_map

        return Image.fromarray(reconstructed.astype(np.uint8))

    def infer(self, img, save_all_timesteps=False, num_timesteps=5, tau=0.01):
        """
        Perform SC-UNSB inference using two-pass Dense Normalization.

        Parameters:
            img (PIL.Image) - input image
            save_all_timesteps (bool) - if True, return all NFE timesteps
            num_timesteps (int) - number of NFE timesteps (default: 5)
            tau (float) - entropy parameter for diffusion process

        Returns:
            PIL.Image or dict of PIL Images - result image(s) at original resolution
        """
        original_size = img.size
        n_h, n_w, patches_info = self.get_patch_grid(original_size)

        print(f'  Grid size: {n_h} x {n_w} = {len(patches_info)} patches')

        # Identify background patches
        bg_patches_mask = []
        patch_tensors = []

        for row, col, left, top, right, bottom in patches_info:
            patch = img.crop((left, top, right, bottom))
            patch_tensor = self.image_to_tensor(patch)
            patch_tensors.append(patch_tensor)

            if not self.disable_bg_detection and self.is_background_patch(patch_tensor):
                bg_patches_mask.append(True)
            else:
                bg_patches_mask.append(False)

        bg_count = sum(bg_patches_mask)
        non_bg_count = len(patches_info) - bg_count
        print(f'  Background patches: {bg_count}/{len(patches_info)} ({100*bg_count/len(patches_info):.1f}%)')
        print(f'  Non-background patches: {non_bg_count}/{len(patches_info)} ({100*non_bg_count/len(patches_info):.1f}%)')

        # ============================================================
        # Pass 1: 标准推理 + 收集颜色统计量
        # ============================================================
        print('  Pass 1: 标准推理并收集颜色统计量...')
        self.model.netG.eval()
        self.model.init_color_stats_collection(n_h, n_w, num_timesteps=num_timesteps)

        # Time schedule for SB
        T = num_timesteps
        incs = np.array([0] + [1/(i+1) for i in range(T-1)])
        times = np.cumsum(incs)
        times = times / times[-1]
        times = 0.5 * times[-1] + 0.5 * times
        times = np.concatenate([np.zeros(1), times])
        times = torch.tensor(times).float()
        if torch.cuda.is_available():
            times = times.cuda()

        # Pass 1: 标准推理并收集每个timestep的颜色统计量
        with torch.no_grad():
            for idx, ((row, col, left, top, right, bottom), patch_tensor, is_bg) in enumerate(
                    zip(patches_info, patch_tensors, bg_patches_mask)):
                if is_bg:
                    continue

                # 运行多步SB前向传播
                Xt = patch_tensor
                for t in range(T):
                    if t > 0:
                        delta = times[t] - times[t-1]
                        denom = times[-1] - times[t-1]
                        inter = (delta / denom).reshape(-1, 1, 1, 1)
                        scale = (delta * (1 - delta / denom)).reshape(-1, 1, 1, 1)
                        Xt = (1 - inter) * Xt + inter * Xt_1.detach() + (scale * tau).sqrt() * torch.randn_like(Xt)

                    time_idx = (t * torch.ones(size=[1])).long()
                    if torch.cuda.is_available():
                        time_idx = time_idx.cuda()
                    z = torch.randn(size=[1, 4 * self.model.opt.ngf])
                    if torch.cuda.is_available():
                        z = z.cuda()

                    # 标准推理（不使用DN）
                    Xt_1 = self.model.netG(Xt, time_idx, z)

                    # 收集该timestep的颜色统计量
                    self.model.collect_patch_color_stats(Xt_1, row, col, t)

        # ============================================================
        # Pass 2: 标准推理 + 颜色插值调整
        # ============================================================
        print('  Pass 2: 标准推理并应用颜色插值调整...')

        # Collect results for each timestep
        timestep_results = {f'fake_{i+1}': [] for i in range(num_timesteps)}

        with torch.no_grad():
            for idx, ((row, col, left, top, right, bottom), patch_tensor, is_bg) in enumerate(
                    zip(patches_info, patch_tensors, bg_patches_mask)):

                if is_bg:
                    # 背景patch：直接复制
                    for t in range(num_timesteps):
                        timestep_results[f'fake_{t+1}'].append(patch_tensor)
                else:
                    # 非背景patch：标准推理 + 颜色插值调整
                    Xt = patch_tensor
                    for t in range(T):
                        if t > 0:
                            delta = times[t] - times[t-1]
                            denom = times[-1] - times[t-1]
                            inter = (delta / denom).reshape(-1, 1, 1, 1)
                            scale = (delta * (1 - delta / denom)).reshape(-1, 1, 1, 1)
                            Xt = (1 - inter) * Xt + inter * Xt_1.detach() + (scale * tau).sqrt() * torch.randn_like(Xt)

                        time_idx = (t * torch.ones(size=[1])).long()
                        if torch.cuda.is_available():
                            time_idx = time_idx.cuda()
                        z = torch.randn(size=[1, 4 * self.model.opt.ngf])
                        if torch.cuda.is_available():
                            z = z.cuda()

                        # 标准推理（不使用DN，保证细胞结构完整）
                        Xt_1 = self.model.netG(Xt, time_idx, z)

                        # 应用颜色插值调整（关键：在每个timestep都进行）
                        Xt_1 = self.model.apply_color_interpolation(
                            Xt_1, row, col, t, padding=self.dn_padding
                        )

                        timestep_results[f'fake_{t+1}'].append(Xt_1)

        # Reconstruct full image for each timestep
        if save_all_timesteps:
            reconstructed_images = {}
            for t in range(num_timesteps):
                timestep_name = f'fake_{t+1}'
                reconstructed_images[timestep_name] = self.reconstruct_image(
                    patches_info, timestep_results[timestep_name], original_size
                )
            return reconstructed_images
        else:
            last_timestep = f'fake_{num_timesteps}'
            reconstructed = self.reconstruct_image(patches_info, timestep_results[last_timestep], original_size)
            return reconstructed


def save_pil_images(webpage, visuals, image_path, width=256):
    """Save PIL/ndarray/tensor visuals to disk and add a row to HTML."""
    image_dir = webpage.get_image_dir()
    short_path = ntpath.basename(image_path)
    name = os.path.splitext(short_path)[0]

    webpage.add_header(name)
    ims, txts, links = [], [], []

    for label, im_data in visuals.items():
        if isinstance(im_data, torch.Tensor):
            im = util.tensor2im(im_data)
        elif isinstance(im_data, Image.Image):
            im = np.array(im_data)
        else:
            im = im_data

        image_name = '%s/%s.png' % (label, name)
        os.makedirs(os.path.join(image_dir, label), exist_ok=True)
        save_path = os.path.join(image_dir, image_name)
        util.save_image(im, save_path)
        ims.append(image_name)
        txts.append(label)
        links.append(image_path)

    webpage.add_images(ims, txts, links, width=width)


def run_sc_unsb_inference(opt):
    """
    Run SC-UNSB inference with Dense Normalization.

    Parameters:
        opt - TestOptions object
    """
    # Setup
    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.display_id = -1

    # Get parameters
    patch_size = getattr(opt, 'patch_size', 256)
    overlap = getattr(opt, 'overlap', 64)
    num_timesteps = getattr(opt, 'num_timesteps', 5)
    bg_threshold = getattr(opt, 'bg_threshold', 0.70)
    disable_bg_detection = getattr(opt, 'disable_bg_detection', False)
    dn_padding = getattr(opt, 'dn_padding', 1)
    tau = getattr(opt, 'tau', 0.01)

    print('SC-UNSB Inference with Dense Normalization:')
    print(f'  patch_size={patch_size}, overlap={overlap}, timesteps={num_timesteps}')
    print(f'  dn_padding={dn_padding}, tau={tau}')
    if not disable_bg_detection:
        print(f'  bg_threshold={bg_threshold}')

    # Create dataset and model
    create_dataset(opt)
    create_dataset(util.copyconf(opt, phase="train"))

    model = create_model(opt)
    model.setup(opt)
    model.parallelize()
    if opt.eval:
        model.eval()

    # Enable plug-and-play DN if model was trained with standard IN
    # This automatically replaces InstanceNorm layers with DenseInstanceNorm
    dn_count = model.enable_dn_plugin()
    if dn_count == 0:
        print('Warning: No DN layers found. SC-UNSB inference may not work correctly.')
        print('Make sure the model has normalization layers.')

    # Get image paths
    from data.image_folder import make_dataset
    test_dir = os.path.join(opt.dataroot, 'allB')
    image_paths = sorted(make_dataset(test_dir, float('inf')))

    if opt.num_test > 0:
        image_paths = image_paths[:opt.num_test]

    print(f'Processing {len(image_paths)} images...')

    # Create webpage
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}_sc_unsb'.format(opt.phase, opt.epoch))
    print('Creating web directory:', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s (SC-UNSB), Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))

    # Create SC-UNSB inferencer
    inferencer = SCUNSBInferencer(
        model,
        patch_size=patch_size,
        overlap=overlap,
        dn_padding=dn_padding,
        bg_threshold=bg_threshold,
        disable_bg_detection=disable_bg_detection
    )

    # Process each image
    for i, img_path in enumerate(tqdm(image_paths, desc='Processing images')):
        try:
            img = Image.open(img_path).convert('RGB')
            original_size = img.size

            print(f'\nProcessing image {i+1}/{len(image_paths)}: {os.path.basename(img_path)} ({original_size})')

            # Run SC-UNSB inference
            all_results = inferencer.infer(
                img,
                save_all_timesteps=True,
                num_timesteps=num_timesteps,
                tau=tau
            )

            # Build visuals
            visuals = OrderedDict()
            visuals['real'] = img
            for t in range(num_timesteps):
                timestep_name = f'fake_{t+1}'
                visuals[timestep_name] = all_results[timestep_name]

            # Save to HTML
            save_pil_images(webpage, visuals, img_path, width=opt.display_winsize)

            # Save each timestep to folders
            img_name = os.path.basename(img_path)
            name_without_ext = os.path.splitext(img_name)[0]
            for t in range(num_timesteps):
                timestep_name = f'fake_{t+1}'
                result_img = all_results[timestep_name]
                save_name = f'{name_without_ext}.png'
                save_dir = os.path.join(web_dir, 'images', timestep_name)
                os.makedirs(save_dir, exist_ok=True)
                result_img.save(os.path.join(save_dir, save_name))

        except Exception as e:
            print(f'Error processing {img_path}: {e}')
            import traceback
            traceback.print_exc()
            continue

    webpage.save()
    print('\nSC-UNSB inference complete! Results saved to:', web_dir)


if __name__ == '__main__':
    opt = TestOptions().parse()
    run_sc_unsb_inference(opt)
