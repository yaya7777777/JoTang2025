"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from model import CNN
from data import create_dataloaders

def analyze_activations():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN().to(device)
    model.load_state_dict(torch.load(os.path.join('..', 'results', 'q1_model.pt'), map_location=device))
    _, val_loader = create_dataloaders()
    
    # 存储激活强度
    layer1_activations = []  # shape: (num_samples, 16)
    layer5_activations = []  # shape: (num_samples, 80)
    labels_list = []
    
    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            # 获取中间特征
            outputs = model(images, intermediate_outputs=True)
            
            # 计算L2范数 (batch_size, channels)
            # outputs返回的是 (final_out, conv1_out, conv2_out, conv3_out, conv4_out, conv5_out)
            conv1_norm = torch.norm(outputs[1], p=2, dim=(2, 3))  # outputs[1]是conv1_out
            conv5_norm = torch.norm(outputs[5], p=2, dim=(2, 3))  # outputs[5]是conv5_out
            
            layer1_activations.append(conv1_norm)
            layer5_activations.append(conv5_norm)
            labels_list.append(labels)
    
    # 合并所有batch的结果
    layer1_activations = torch.cat(layer1_activations, dim=0).cpu().numpy()
    layer5_activations = torch.cat(layer5_activations, dim=0).cpu().numpy()
    labels = torch.cat(labels_list, dim=0).cpu().numpy()
    
    # 计算类别平均激活强度
    class_names = [str(i) for i in range(10)]
    # 热图可视化
    plot_heatmaps(layer1_activations, labels, class_names, "Layer 1", "q3_layer1_heatmap.pdf")
    plot_heatmaps(layer5_activations, labels, class_names, "Layer 5", "q3_layer5_heatmap.pdf")
    
    # 卷积核可视化
    plot_all_kernels(layer1_activations, labels, class_names, "Layer 1", "q3_layer1_kernels.png")
    plot_all_kernels(layer5_activations, labels, class_names, "Layer 5", "q3_layer5_kernels.png")


def plot_heatmaps(activations, labels, class_names, layer_name, save_path):
    # 计算每个类别每个卷积核的平均激活强度
    num_classes = len(class_names)
    num_filters = activations.shape[1]
    
    avg_activations = np.zeros((num_classes, num_filters))
    for c in range(num_classes):
        class_mask = (labels == c)
        avg_activations[c] = activations[class_mask].mean(axis=0)
    
    # 归一化
    avg_activations = (avg_activations - avg_activations.min()) / \
                      (avg_activations.max() - avg_activations.min())
    
    # 创建热图
    plt.figure(figsize=(20, 8) if layer_name == "Layer 5" else (12, 8))
    plt.imshow(avg_activations.T, cmap='viridis', aspect='auto')
    plt.colorbar(label='Normalized Activation Strength')
    
    # 设置坐标轴
    plt.yticks(range(num_filters), [f'Filter {i+1}' for i in range(num_filters)])
    plt.xticks(range(num_classes), class_names)
    
    plt.title(f'{layer_name} - Heatmap of Average Activation Strength per Class')
    plt.xlabel('Class')
    plt.ylabel('Convolutional Filter')
    
    plt.tight_layout()
    plt.savefig(os.path.join('..', 'results', save_path))
    plt.close()


def plot_all_kernels(activations, labels, class_names, title, save_path):
    
    # 计算每个类别每个卷积核的平均激活强度
    num_classes = len(class_names)
    num_kernels = activations.shape[1]
    
    avg_activations = np.zeros((num_kernels, num_classes))
    for c in range(num_classes):
        class_mask = (labels == c)
        avg_activations[:, c] = activations[class_mask].mean(axis=0)
    
    # 归一化
    avg_activations = (avg_activations - avg_activations.min()) / \
                      (avg_activations.max() - avg_activations.min())
    
    
    # 创建大图
    plt.figure(figsize=(16, num_kernels*0.8))
    for i in range(num_kernels):
        plt.subplot(num_kernels, 1, i+1)
        bars = plt.bar(range(num_classes), avg_activations[i], color=plt.cm.viridis(i/num_kernels))
        
        # Set plot properties
        plt.title(f'Kernel {i+1}', fontsize=10)
        plt.xlabel('Class')
        plt.ylabel('Activation Strength')
        plt.ylim(0, 1.1)
        plt.grid(True, alpha=0.3)
        plt.xticks(range(num_classes), class_names)
        
        # 在每个柱子上标注数值
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom')
    plt.suptitle(title, y=1.02, fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join('..', 'results', save_path), dpi=300)
    plt.close()

if __name__ == '__main__':
    analyze_activations()
    
"""