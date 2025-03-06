import os
import argparse
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from model import TEN  # 假設 model.py 定義了 TEN

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

def enhance_image(model, image_path, device):
    """
    讀取 image_path 的影像，轉成灰階並增強
    """
    img = Image.open(image_path).convert("L")
    transform = transforms.ToTensor()
    img_tensor = transform(img).unsqueeze(0)  # 加上 batch 維度
    model.eval()
    with torch.no_grad():
        output = model(img_tensor.to(device))
    # 輸出為 numpy array (單通道)
    return output.squeeze().cpu().numpy()

def main(args):
    # 設定設備
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 建立模型並載入權重
    model = TEN().to(device)
    checkpoint_path = "checkpoints/ten_best.pth"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"找不到 checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    # 移除 'module.' 前綴
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        name = k.replace("module.", "")  # 去除 "module." 前綴
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    
    # 取得輸出資料夾
    output_folder = get_next_exp_folder(args.output_root)
    os.makedirs(output_folder, exist_ok=True)
    print(f"輸出資料夾：{output_folder}")
    
    # 取得輸入資料夾中所有影像檔案（支援 jpg, png, jpeg, bmp, tif, tiff）
    valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    input_folder = args.input_folder
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(valid_exts)]
    if not image_files:
        print("輸入資料夾中沒有找到影像檔案。")
        return
    
    # 對每個影像進行增強，並儲存結果
    for fname in image_files:
        in_path = os.path.join(input_folder, fname)
        print(f"處理影像: {in_path}")
        enhanced = enhance_image(model, in_path, device)
        
        # 將 numpy 轉換為 PIL Image 進行存檔
        out_img = Image.fromarray((enhanced * 255).astype('uint8'))
        out_path = os.path.join(output_folder, fname)
        out_img.save(out_path)
    
    print("所有影像已增強完成！")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="利用 TEN 模型對指定路徑下所有影像做增強")
    parser.add_argument("--input_folder", type=str, required=True, help="待增強影像所在資料夾路徑")
    parser.add_argument("--output_root", type=str, default="./output", help="輸出根資料夾路徑 (預設 ./output)")
    
    args = parser.parse_args()
    main(args)
