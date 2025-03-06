import os
import random
from PIL import Image, ImageFilter
import torchvision.transforms as transforms
from torch.utils.data import Dataset

class ThermalDataset(Dataset):
    def __init__(self, image_dir, patch_size=36, scale=2, stride=6, transform=None, sliding=True):
        """
        image_dir: 圖片文件夾路徑
        patch_size: 裁剪 patch 的尺寸 (預設36)
        scale: 放大倍率 (例如2或3)
        stride: 滑動視窗的步長 (例如6)
        transform: 可選的圖像變換
        sliding: 若為 True 則採用滑動視窗切割整張影像；若 False 則隨機切一個 patch
        """
        self.image_dir = image_dir
        self.patch_size = patch_size
        self.scale = scale
        self.stride = stride
        self.transform = transform
        self.sliding = sliding
        self.image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir)
                            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # 加載圖片並轉換為灰度圖
        img = Image.open(self.image_paths[idx]).convert('L')
        width, height = img.size
        # 若影像尺寸過小，先做調整
        if width < self.patch_size or height < self.patch_size:
            img = img.resize((max(width, self.patch_size), max(height, self.patch_size)), Image.BICUBIC)
            width, height = img.size
        
        patches_lr = []
        patches_gt = []
        
        if self.sliding:
            # 使用滑動視窗方式，依據 stride 切出所有 patch
            for top in range(0, height - self.patch_size + 1, self.stride):
                for left in range(0, width - self.patch_size + 1, self.stride):
                    crop = img.crop((left, top, left + self.patch_size, top + self.patch_size))
                    gt = crop  # 高解析度 patch
                    # 產生低解析度 patch
                    sigma = 1.0 if self.scale == 2 else 1.5
                    lr = crop.filter(ImageFilter.GaussianBlur(radius=sigma))
                    lr_size = (self.patch_size // self.scale, self.patch_size // self.scale)
                    lr = lr.resize(lr_size, Image.BICUBIC)
                    lr = lr.resize((self.patch_size, self.patch_size), Image.BICUBIC)
                    
                    # 應用變換（如果有設定 transform）
                    if self.transform:
                        gt = self.transform(gt)
                        lr = self.transform(lr)
                    else:
                        to_tensor = transforms.ToTensor()
                        gt = to_tensor(gt)
                        lr = to_tensor(lr)
                    
                    patches_gt.append(gt)
                    patches_lr.append(lr)
            return patches_lr, patches_gt
        else:
            # 原本的隨機裁剪做法，只返回一個 patch
            left = random.randint(0, width - self.patch_size)
            top = random.randint(0, height - self.patch_size)
            crop = img.crop((left, top, left + self.patch_size, top + self.patch_size))
            gt = crop
            sigma = 1.0 if self.scale == 2 else 1.5
            lr = crop.filter(ImageFilter.GaussianBlur(radius=sigma))
            lr_size = (self.patch_size // self.scale, self.patch_size // self.scale)
            lr = lr.resize(lr_size, Image.BICUBIC)
            lr = lr.resize((self.patch_size, self.patch_size), Image.BICUBIC)
            if self.transform:
                gt = self.transform(gt)
                lr = self.transform(lr)
            else:
                to_tensor = transforms.ToTensor()
                gt = to_tensor(gt)
                lr = to_tensor(lr)
            return lr, gt

class FullImageThermalDataset(Dataset):
    def __init__(self, image_dir, scale=2, transform=None, resize=None):
        """
        image_dir: 影像資料夾路徑
        scale: 放大倍率 (例如2或3)
        transform: 可選的圖像變換
        resize: 如果需要，將整張影像調整為指定尺寸，格式為 (width, height)
        """
        self.image_dir = image_dir
        self.scale = scale
        self.transform = transform
        self.resize = resize
        self.image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir)
                            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # 加載影像並轉成灰階
        img = Image.open(self.image_paths[idx]).convert('L')
        if self.resize:
            img = img.resize(self.resize, Image.BICUBIC)
        
        # 高解析度影像
        gt = img
        
        # 生成低解析度影像：用 Gaussian 模糊、下採樣再上採樣到原尺寸
        sigma = 1.0 if self.scale == 2 else 1.5
        lr = img.filter(ImageFilter.GaussianBlur(radius=sigma))
        
        # 下採樣
        width, height = img.size
        lr_size = (width // self.scale, height // self.scale)
        lr = lr.resize(lr_size, Image.BICUBIC)
        # 再上採樣回原尺寸
        lr = lr.resize((width, height), Image.BICUBIC)
        
        # 應用變換（例如 ToTensor）
        if self.transform:
            gt = self.transform(gt)
            lr = self.transform(lr)
        else:
            to_tensor = transforms.ToTensor()
            gt = to_tensor(gt)
            lr = to_tensor(lr)
        
        return lr, gt