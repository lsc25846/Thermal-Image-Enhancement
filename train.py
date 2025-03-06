import os
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import argparse

from model import TEN
from dataset import ThermalDataset, FullImageThermalDataset

def parse_resize(resize_str):
    """解析 --resize 參數，格式必須為 'width,height'，例如 '320,240'"""
    try:
        width, height = map(int, resize_str.split(','))
        return (width, height)
    except Exception as e:
        raise argparse.ArgumentTypeError("resize 參數格式必須為 'width,height'，例如 '320,240'") from e

def get_next_exp_folder(output_root="./output"):
    """
    搜尋 output_root 下所有 exp 資料夾，返回最高數字 + 1 的資料夾名稱
    """
    if not os.path.exists(output_root):
        os.makedirs(output_root)
        return os.path.join(output_root, "exp0")
    
    exp_nums = []
    for folder in os.listdir(output_root):
        if folder.startswith("exp"):
            try:
                num = int(folder[3:])
                exp_nums.append(num)
            except ValueError:
                pass
    next_exp = max(exp_nums) + 1 if exp_nums else 0
    return os.path.join(output_root, f"exp{next_exp}")

def main():
    parser = argparse.ArgumentParser(description="訓練 TEN 模型，可選用 patch 或 full 資料集")
    # 資料集相關參數
    parser.add_argument("--dataset_type", type=str, choices=["patch", "full"], default="patch",
                        help="選擇資料集類型：'patch' (ThermalDataset) 或 'full' (FullImageThermalDataset)")
    parser.add_argument("--train_dir", type=str, default="/mnt/e/desktop/TEN_thermal/train",
                        help="訓練資料夾路徑")
    parser.add_argument("--val_dir", type=str, default="/mnt/e/desktop/TEN_thermal/val",
                        help="驗證資料夾路徑")
    parser.add_argument("--resize", type=parse_resize, default=None,
                        help="(僅 full 模式) 以 'width,height' 格式指定整張影像的尺寸，例如 '320,240'")
    
    # 訓練相關參數
    parser.add_argument("--patch_size", type=int, default=64, help="patch 尺寸 (僅在 patch 模式下生效)")
    parser.add_argument("--scale", type=int, default=2, help="放大倍率 (可換成 3)")
    parser.add_argument("--batch_size", type=int, default=1024, help="batch size")
    parser.add_argument("--num_epochs", type=int, default=1700, help="訓練 epoch 數")
    parser.add_argument("--initial_lr", type=float, default=0.001, help="初始學習率")
    
    # GPU 與預訓練參數
    parser.add_argument("--device", type=str, default="cuda", help="訓練裝置，例如 'cuda:0' 或 'cpu'")
    parser.add_argument("--pretrain", type=str, default=None, help="預訓練權重檔案路徑，若提供則載入該權重")
    parser.add_argument("--output_root", type=str, default="./output", help="checkpoint 與 log 輸出的根目錄")
    
    args = parser.parse_args()

    # 建立設備
    device = torch.device(args.device if torch.cuda.is_available() or args.device=="cpu" else "cpu")

    # 取得新的 exp 輸出資料夾
    output_folder = get_next_exp_folder(args.output_root)
    os.makedirs(output_folder, exist_ok=True)
    print(f"輸出資料夾：{output_folder}")

    # 根據 dataset_type 建立資料集
    if args.dataset_type == "patch":
        print("使用 ThermalDataset (隨機裁剪 patch) 模式")
        train_dataset = ThermalDataset(
            image_dir=args.train_dir,
            patch_size=args.patch_size,
            scale=args.scale,
            transform=None,
            sliding=False  # 使用隨機裁剪
        )
        val_dataset = ThermalDataset(
            image_dir=args.val_dir,
            patch_size=args.patch_size,
            scale=args.scale,
            transform=None,
            sliding=False
        )
    else:
        print("使用 FullImageThermalDataset (整張影像) 模式")
        train_dataset = FullImageThermalDataset(
            image_dir=args.train_dir,
            scale=args.scale,
            transform=transforms.ToTensor(),
            resize=args.resize
        )
        val_dataset = FullImageThermalDataset(
            image_dir=args.val_dir,
            scale=args.scale,
            transform=transforms.ToTensor(),
            resize=args.resize
        )

    # DataLoader 參數
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16)
    val_loader   = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=16)

    # 建立模型並搬移到裝置上
    model = TEN().to(device)
    model = torch.nn.DataParallel(model)

    # 載入預訓練權重（如果有指定）
    if args.pretrain is not None:
        if os.path.exists(args.pretrain):
            print(f"載入預訓練權重：{args.pretrain}")
            checkpoint = torch.load(args.pretrain, map_location=device)
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in checkpoint.items():
                new_state_dict[k.replace("module.", "")] = v
            model.load_state_dict(new_state_dict)
        else:
            print(f"預訓練權重檔案 {args.pretrain} 不存在，將從頭訓練。")

    # 設定損失函數、優化器與學習率調度
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=args.initial_lr, momentum=0.9, weight_decay=0.0005)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)

    best_val_loss = float('inf')
    
    # 準備 log 檔案
    log_path = os.path.join(output_folder, "loss_log.txt")
    log_file = open(log_path, "w")
    log_file.write("epoch,train_loss,val_loss,epoch_time(sec)\n")

    os.makedirs("sample", exist_ok=True)  # 儲存 sample 結果的資料夾

    # 訓練迴圈
    for epoch in range(1, args.num_epochs + 1):
        start_time = time.time()

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss = validate(model, val_loader, criterion, device)

        epoch_time = time.time() - start_time
        log_line = f"{epoch},{train_loss:.6f},{val_loss:.6f},{epoch_time:.2f}\n"
        log_file.write(log_line)
        log_file.flush()

        print(f"[Epoch {epoch}/{args.num_epochs}] Val Loss: {val_loss:.6f} | Epoch Time: {epoch_time:.2f} 秒")

        sample_inference(model, val_dataset, device, epoch, n_samples=4, patch_size=args.patch_size)
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = os.path.join(output_folder, "ten_best.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Best checkpoint saved to {ckpt_path}")

    log_file.close()
    print("訓練完成。")

def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    for batch_idx, (lr, gt) in enumerate(dataloader):
        lr, gt = lr.to(device), gt.to(device)
        optimizer.zero_grad()
        output = model(lr)
        loss = criterion(output, gt)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * lr.size(0)
        # 如有需要可啟用以下 batch loss 輸出：
        # print(f"[Epoch {epoch}][Batch {batch_idx}] Loss: {loss.item():.6f}")
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for lr, gt in dataloader:
            lr, gt = lr.to(device), gt.to(device)
            output = model(lr)
            loss = criterion(output, gt)
            running_loss += loss.item() * lr.size(0)
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

def sample_inference(model, val_dataset, device, epoch, n_samples=4, patch_size=64):
    import random
    indices = random.sample(range(len(val_dataset)), n_samples)
    lr_list = []
    pred_list = []
    gt_list = []
    model.eval()
    with torch.no_grad():
        for idx in indices:
            data = val_dataset[idx]
            if isinstance(data[0], list):
                list_lr = data[0]
                list_gt = data[1]
                sub_idx = random.randint(0, len(list_lr)-1)
                lr_tensor = list_lr[sub_idx].unsqueeze(0).to(device)
                gt_tensor = list_gt[sub_idx].unsqueeze(0).to(device)
            else:
                lr_tensor = data[0].unsqueeze(0).to(device)
                gt_tensor = data[1].unsqueeze(0).to(device)
            pred = model(lr_tensor)
            lr_np   = lr_tensor.squeeze().cpu().numpy()
            pred_np = pred.squeeze().cpu().numpy()
            gt_np   = gt_tensor.squeeze().cpu().numpy()
            lr_list.append(lr_np)
            pred_list.append(pred_np)
            gt_list.append(gt_np)
    fig, axes = plt.subplots(nrows=3, ncols=n_samples, figsize=(3*n_samples, 6))
    for col in range(n_samples):
        axes[0, col].imshow(lr_list[col], cmap='gray')
        axes[0, col].axis('off')
        axes[0, col].set_title(f"LR {col}")
        axes[1, col].imshow(pred_list[col], cmap='gray')
        axes[1, col].axis('off')
        axes[1, col].set_title(f"Pred {col}")
        axes[2, col].imshow(gt_list[col], cmap='gray')
        axes[2, col].axis('off')
        axes[2, col].set_title(f"GT {col}")
    plt.tight_layout()
    save_path = f"./sample/epoch_{epoch}_sample.png"
    plt.savefig(save_path)
    plt.close()
    print(f"Sample results saved to {save_path}")

if __name__ == "__main__":
    main()
