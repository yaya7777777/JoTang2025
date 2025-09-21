import torch
import torch
import matplotlib.pyplot as plt
import os

from data import create_dataloaders
from model import CNN

if __name__ == '__main__':
    model = CNN()
    train_loader, test_loader = create_dataloaders()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载模型参数
    model.load_state_dict(torch.load(os.path.join('..', 'results', 'q1_model.pt'), weights_only=True))
    model.to(device)
    
    model.eval()
    
    # 创建保存目录
    output_dir = os.path.join('..', 'results', "q3_filters")
    os.makedirs(output_dir, exist_ok=True)
    
    # 每层卷积的输出通道数
    num_fs = [16, 32, 48, 64, 80]
    norms = torch.zeros(len(test_loader.dataset), sum(num_fs))
    all_labels = torch.zeros(len(test_loader.dataset))
    
    step = 0
    for images, labels in test_loader:
        # 将数据移至正确设备
        images = images.to(device)
        labels = labels.to(device)
        
        # 前向传播，获取中间特征
        _, x1, x2, x3, x4, x5 = model(images, intermediate_outputs=True)
        
        # 计算中间输出在空间维度上的范数（norm）
        x1_norm = torch.norm(x1.view(x1.size(0), x1.size(1), -1), p=2, dim=2)
        x2_norm = torch.norm(x2.view(x2.size(0), x2.size(1), -1), p=2, dim=2)
        x3_norm = torch.norm(x3.view(x3.size(0), x3.size(1), -1), p=2, dim=2)
        x4_norm = torch.norm(x4.view(x4.size(0), x4.size(1), -1), p=2, dim=2)
        x5_norm = torch.norm(x5.view(x5.size(0), x5.size(1), -1), p=2, dim=2)
            
        # 将范数值写入 norms 张量
        f_idx = 0
        norms[step:step+images.size(0), f_idx:f_idx+num_fs[0]] = x1_norm.detach().cpu()
        f_idx += num_fs[0]
        norms[step:step+images.size(0), f_idx:f_idx+num_fs[1]] = x2_norm.detach().cpu()
        f_idx += num_fs[1]
        norms[step:step+images.size(0), f_idx:f_idx+num_fs[2]] = x3_norm.detach().cpu()
        f_idx += num_fs[2]
        norms[step:step+images.size(0), f_idx:f_idx+num_fs[3]] = x4_norm.detach().cpu()
        f_idx += num_fs[3]
        norms[step:step+images.size(0), f_idx:f_idx+num_fs[4]] = x5_norm.detach().cpu()
        
        # 保存 labels
        all_labels[step:step+images.size(0)] = labels
        
        step += images.size(0)
    
    labelnames = ["airplane", "automobile", "bird", "cat", "deer",
                  "dog", "frog", "horse", "ship", "truck"]
    
    # 按类别计算平均激活强度
    classwise_score_avg = torch.zeros(10, sum(num_fs))
    for l in range(10):
        classwise_score_avg[l] = norms[all_labels == l].mean(dim=0)
    
    # 为每个卷积核生成类别平均得分柱状图
    start = 0
    for layer_idx, num_f in enumerate(num_fs):
        layer_dir = os.path.join(output_dir, f"classwise_avg_{layer_idx}")
        os.makedirs(layer_dir, exist_ok=True)
        
        for f_idx in range(num_f):
            fig, ax = plt.subplots()
            data = classwise_score_avg[:, start+f_idx]
            data /= data.max()  # 归一化
            ax.bar(labelnames, data)
            ax.set_title(f"Filter {f_idx}")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(layer_dir, f"filter_{f_idx}.png"))
            plt.close()
        
        start += num_f
