import torch
import matplotlib.pyplot as plt
torch.manual_seed(123)

from data import create_dataloaders
from model import CNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader, val_loader = create_dataloaders()

model = CNN()
model.to(device)

# TODO: 设置你的 training parameters
num_epochs = None
lr = None
weight_decay = None

# TODO: 设置你的 cross-entropy loss function
loss_fn = None

# TODO: 设置你的优化器，注意用上你的 lr 和 weight_decay
optimizer = None

# TODO: 设置你的 learning rate scheduler
scheduler = None

# 用于画图
train_loss_list = []
train_accuracy_list = []
val_loss_list = []
val_accuracy_list = []
epoch_list = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for i, (images, labels) in enumerate(train_loader):
        # TODO: 将图像和标签移动到设备上

        # TODO: 将梯度清零

        # TODO: 通过模型进行前向传播

        # TODO: 计算损失
        loss = None

        # TODO: 反向传播

        # TODO: 更新权重

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
            outputs = None
            loss = None

            # TODO: 从模型输出中获取预测标签 label
            predicted = None

            # TODO: 累加 correct 和 total


        epoch_val_accuracy = correct / total
        epoch_val_loss /= total
        val_loss_list.append(epoch_val_loss)
        val_accuracy_list.append(epoch_val_accuracy)

        epoch_list.append(epoch)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_accuracy:.4f}")

        # TODO: （可选）在这里，你可以保存效果最好的模型


# 如果你之前没有保存模型，这里会保存最后一轮的模型状态
torch.save(model.state_dict(), "q1_model.pt")

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
plt.savefig(f"q1_plots.png", dpi=300)
plt.clf()
plt.close()




