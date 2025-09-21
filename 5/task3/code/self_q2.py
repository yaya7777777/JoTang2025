"""
import os
import matplotlib.pyplot as plt
import torch
from model import CNN

def visualize_first_layer():
    # 加载训练模型
    model = CNN()
    model.load_state_dict(torch.load(os.path.join('..', 'results', 'q1_model.pt'), map_location=torch.device('cpu')))
    
    # 获取第一层卷积核权重
    conv1 = model.conv1.weight.data.cpu()
    
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    fig.suptitle('First Layer Convolutional Filters')
    
    
    # 绘制每个卷积核
    for i, ax in enumerate(axes.flat):# 平展为一维，便于遍历
        if i < conv1.shape[0]:  # 确保不超过16个卷积核
            # 归一化
            kernel = conv1[i].permute(1, 2, 0)# 转化至imshow格式
            kernel = (kernel - kernel.min()) / (kernel.max() - kernel.min())
            ax.imshow(kernel)
            ax.set_title(f'Filter {i+1}')
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join('..', 'results', 'q2_filters.png'))
    plt.close()
   

if __name__ == '__main__':
    visualize_first_layer()
"""