import torch
import os
from PIL import Image
from glob import glob
from model import CNN

from torchvision import transforms



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN()
model.to(device)
model_path = os.path.join("task3", "results", "q1_model.pt")
model.load_state_dict(torch.load(model_path, weights_only=True))
model.eval()

test_images_path = os.path.join("task3", "data", "test_unlabeled", "*.png")
test_images = sorted(glob(test_images_path))

# TODO: 创建测试时的图像 transformations.
test_tf = transforms.Compose([
    transforms.Resize((32, 32)),# 调整图像大小为32x32
    transforms.ToTensor(),# 转换为pytorch张量
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))# 标准化
    
])

test_result_path = os.path.join("task3", "results", "q1_test.txt")
test_write = open(test_result_path, "w")
for imgfile in test_images:
    filename = os.path.basename(imgfile)
    img = Image.open(imgfile)
    img = test_tf(img)
    img = img.unsqueeze(0).to(device)

    # TODO: 使模型进行前向传播并获取预测标签，predicted 是一个 PyTorch 张量，包含预测的标签，值为 0 到 9 之间的单个整数（包含 0 和 9）
    with torch.no_grad():
        output = model(img)
        _, predicted = torch.max(output, 1)
    
    

    test_write.write(f"{filename},{predicted.item()}\n")
test_write.close()