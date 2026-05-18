# 核心配置
REAL_IMAGES_DIR = r"E:\0_All_Dataset\02cell\A\test\images"
FAKE_IMAGES_DIR = r"E:\细胞domain\UNSB-main\exp3-V2\cell_v2\test_latest_v2_patch\images\fake_5"
DEVICE = "cuda"
BATCH_SIZE = 16
RANDOM_SEED = 42

import os
import warnings
warnings.filterwarnings("ignore")
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms
from torchvision.models import inception_v3, Inception_V3_Weights
from scipy import linalg

# 固定随机种子
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
torch.backends.cudnn.deterministic = True

# 设置PyTorch缓存路径
os.environ['TORCH_HOME'] = r"E:\细胞domain\torch_cache"
os.makedirs(os.environ['TORCH_HOME'], exist_ok=True)

class FIDKIDCalculator:
    def __init__(self):
        # 加载Inception V3模型
        self.device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
        weights = Inception_V3_Weights.IMAGENET1K_V1
        self.model = inception_v3(weights=weights, transform_input=False)
        self.model.fc = torch.nn.Identity()
        self.model.eval().to(self.device)

        # 无变形预处理（等比缩放+居中裁剪）
        self.transform = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def _extract_features(self, img_dir):
        """提取特征（带异常防护）"""
        img_paths = list(Path(img_dir).glob("**/*.jpg")) + \
                    list(Path(img_dir).glob("**/*.png")) + \
                    list(Path(img_dir).glob("**/*.bmp"))
        if not img_paths:
            raise ValueError(f"路径 {img_dir} 未找到图像文件")

        features = []
        with torch.no_grad():
            for i in range(0, len(img_paths), BATCH_SIZE):
                batch = []
                for p in img_paths[i:i+BATCH_SIZE]:
                    try:
                        img = Image.open(str(p)).convert("RGB")
                        batch.append(self.transform(img))
                    except:
                        continue
                if not batch:
                    continue
                batch = torch.stack(batch).to(self.device)
                features.append(self.model(batch).cpu().numpy())
        return np.concatenate(features, axis=0)

    def calculate_fid(self, feat_real, feat_fake):
        """计算FID（带数值保护）"""
        mu_r, sigma_r = feat_real.mean(axis=0), np.cov(feat_real, rowvar=False)
        mu_f, sigma_f = feat_fake.mean(axis=0), np.cov(feat_fake, rowvar=False)
        
        # 防止奇异矩阵
        eps = 1e-6
        sigma_r += eps * np.eye(sigma_r.shape[0])
        sigma_f += eps * np.eye(sigma_f.shape[0])
        
        diff = mu_r - mu_f
        covmean = linalg.sqrtm(sigma_r @ sigma_f, disp=False)[0].real
        fid = np.sum(diff**2) + np.trace(sigma_r) + np.trace(sigma_f) - 2 * np.trace(covmean)
        return max(float(fid), 0.0)

    def calculate_kid(self, feat_real, feat_fake):
        """计算KID（解决nan问题，带异常防护）"""
        # 采样适配（避免样本数不足导致nan）
        sample_size = min(500, len(feat_real), len(feat_fake))
        if sample_size < 2:
            sample_size = min(len(feat_real), len(feat_fake))
        if sample_size < 2:
            return 0.0  # 样本过少返回0，避免计算错误

        # 固定采样
        idx_r = np.random.choice(len(feat_real), sample_size, replace=False)
        idx_f = np.random.choice(len(feat_fake), sample_size, replace=False)
        fr, ff = feat_real[idx_r], feat_fake[idx_f]

        # RBF核（自适应sigma，避免nan）
        dist = linalg.norm(fr[:, None] - ff[None, :], axis=2)
        sigma = np.median(dist) if np.median(dist) > 0 else 1.0
        k_r = np.exp(-(np.sum(fr**2, 1)[:, None] + np.sum(fr**2, 1) - 2*fr @ fr.T) / (2*sigma**2))
        k_f = np.exp(-(np.sum(ff**2, 1)[:, None] + np.sum(ff**2, 1) - 2*ff @ ff.T) / (2*sigma**2))
        k_rf = np.exp(-(np.sum(fr**2, 1)[:, None] + np.sum(ff**2, 1) - 2*fr @ ff.T) / (2*sigma**2))

        # 排除自相关+防止nan
        np.fill_diagonal(k_r, 0)
        np.fill_diagonal(k_f, 0)
        kid = np.mean(k_r) + np.mean(k_f) - 2 * np.mean(k_rf)
        return float(np.sqrt(max(kid, 1e-8)))  # 避免负数开方导致nan

    def run(self):
        """主计算流程"""
        # 提取特征
        feat_r = self._extract_features(REAL_IMAGES_DIR)
        feat_f = self._extract_features(FAKE_IMAGES_DIR)
        
        # 计算指标
        fid_score = self.calculate_fid(feat_r, feat_f)
        kid_score = self.calculate_kid(feat_r, feat_f)
        
        # 仅输出核心结果
        print(f"FID: {fid_score:.4f}")
        print(f"KID: {kid_score:.4f}")

if __name__ == "__main__":
    try:
        calculator = FIDKIDCalculator()
        calculator.run()
    except Exception as e:
        print(f"计算出错: {str(e)}")