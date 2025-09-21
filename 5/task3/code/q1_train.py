
import os
import torch
import matplotlib.pyplot as plt
from data import create_dataloaders
from model import CNN
import torch.nn as nn

if __name__ == '__main__':
    torch.manual_seed(123)
    
    # 获取当前文件所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 构建结果目录的绝对路径
    results_dir = os.path.join(current_dir, '..', 'results')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = create_dataloaders()
    
    model = CNN()
    model.to(device)
    
    # TODO: 设置你的 training parameters
    num_epochs = 15
    lr = 0.002
    weight_decay = 1e-4
    
    # TODO: 设置你的 cross-entropy loss function
    loss_fn = nn.CrossEntropyLoss()
    
    # TODO: 设置你的优化器，注意用上你的 lr 和 weight_decay
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # TODO: 设置你的 learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    # 用于画图
    train_loss_list = []
    train_accuracy_list = []
    val_loss_list = []
    val_accuracy_list = []
    epoch_list = []
    
    # 用于保存最佳模型
    best_val_accuracy = 0.0
    best_model_state = None
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for i, (images, labels) in enumerate(train_loader):
            # TODO: 将图像和标签移动到设备上
            images = images.to(device)
            labels = labels.to(device)
            # TODO: 将梯度清零
            optimizer.zero_grad()
            # TODO: 通过模型进行前向传播
            outputs = model(images)
            # TODO: 计算损失
            loss = loss_fn(outputs, labels)
            # TODO: 反向传播
            loss.backward()
            # TODO: 更新权重
            optimizer.step()
            
            
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        # 计算训练集每个 Epoch 的损失和准确率
        epoch_train_loss = running_loss / total
        epoch_train_acc = correct / total
        train_loss_list.append(epoch_train_loss)
        train_accuracy_list.append(epoch_train_acc)
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}")

        # 更新学习率：这里每个 epoch 调用一次 step，如果使用 per-batch 调度器，需要放到训练循环内部每个 batch 后
        scheduler.step()
        
        model.eval()
        with torch.no_grad():
            # Compute validation loss and accuracy
            correct, total = 0, 0
            epoch_val_loss = 0.
            for images, labels in val_loader:
                # TODO: 向前传播
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = loss_fn(outputs, labels)

                # TODO: 从模型输出中获取预测标签 label
                _, predicted = torch.max(outputs, 1)

                # TODO: 累加 correct 和 total
                epoch_val_loss += loss.item() * images.size(0)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)


            epoch_val_accuracy = correct / total
            epoch_val_loss /= total
            val_loss_list.append(epoch_val_loss)
            val_accuracy_list.append(epoch_val_accuracy)

            epoch_list.append(epoch)

            print(f"Epoch [{epoch + 1}/{num_epochs}], Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_accuracy:.4f}")

             # TODO: （可选）在这里，你可以保存效果最好的模型
            if epoch_val_accuracy > best_val_accuracy:
                best_val_accuracy = epoch_val_accuracy
                best_model_state = model.state_dict()


    # 如果你之前没有保存模型，这里会保存最后一轮的模型状态

    if best_model_state is not None:
        torch.save(best_model_state, os.path.join(results_dir, 'q1_model.pt'))
        print(f"Saved best model with validation accuracy: {best_val_accuracy:.4f}")
    else:
        torch.save(model.state_dict(), os.path.join(results_dir, 'q1_model.pt'))
        print("Saved final model")
    
    # 绘制 training 和 validation 的 loss 和 accuracy 曲线
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    axs[0].plot(epoch_list, train_loss_list, label="Train")
    axs[0].plot(epoch_list, val_loss_list, label="Val")
    axs[0].set_yscale("log")
    
    axs[1].plot(epoch_list, train_accuracy_list, label="Train")
    axs[1].plot(epoch_list, val_accuracy_list, label="Val")
    
    axs[0].set_title("Loss")
    axs[1].set_title("Accuracy")
    
    for ax in axs:
        ax.legend()
        ax.grid()
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Value")
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'q1_plots.png'), dpi=300)
    plt.clf()
    plt.close()




