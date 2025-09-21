import torch
import imageio.v2 as imio
import os
import numpy as np
from PIL import Image

from model import CNN

model = CNN()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型参数
model.load_state_dict(torch.load(os.path.join('..', 'results', 'q1_model.pt'), weights_only=True))

model.eval()

# TODO：获取 conv1 layer 的权重
conv_weights = model.conv1.weight.data

# 创建保存目录
output_dir = os.path.join('..', 'results', "q2_filters")
os.makedirs(output_dir, exist_ok=True)

for i in range(conv_weights.shape[0]):
     # TODO: 获取第 i 个卷积核
    f = conv_weights[i].cpu().numpy()
    
    # 将3通道卷积核转换为灰度图
    f_gray = np.mean(f, axis=0)
    
    # 将卷积核归一化到 [0, 255] 并转换为 uint8 类型
    f_min = f_gray.min()
    f_max = f_gray.max()
    f_gray = ((f_gray - f_min) / (f_max - f_min) * 255).astype('uint8')
    
    # 创建PIL图像对象
    img = Image.fromarray(f_gray)
    
    # 调整图片大小（放大到合适尺寸）
    scale_factor = 10  # 放大倍数
    new_size = (img.width * scale_factor, img.height * scale_factor)
    img_resized = img.resize(new_size, Image.Resampling.NEAREST)
    
    # 保存为图片
    img_resized.save(os.path.join(output_dir, f"filter_{i}.png"))
