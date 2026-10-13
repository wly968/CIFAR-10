import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets,transforms
from torch.utils.data import DataLoader

#====================超参数====================
batch_size = 100
epoch = 100
learning_rate = 0.001
weight_decay = 5e-4
device = torch.device("cuda"if torch.cuda.is_available()else "cpu")

MEAN = (0.4914, 0.4822, 0.4465)
STD  = (0.2470, 0.2435, 0.2616)

#====================数据加载====================
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),#随机裁剪
    transforms.RandomHorizontalFlip(),#随机水平翻转
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),#颜色抖动
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
    transforms.RandomErasing(p=0.25),#随机遮挡（可选，去掉也能到 94%）
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

train_dataset = datasets.CIFAR10(root="./data",train=True,download=True,transform=train_transform)
test_dataset = datasets.CIFAR10(root="./data",train=False,download=True,transform=test_transform)

train_dataloader = DataLoader(train_dataset,batch_size,shuffle=True,num_workers=2,pin_memory=True)
test_dataloader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False,num_workers=2,pin_memory=True)

#====================模型====================
class BasicBlock(nn.Module):
    """ResNet 的基本残差块：两层 3x3 卷积 + 跳跃连接"""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # 如果通道数或尺寸变了，跳跃连接需要用 1x1 卷积对齐
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)   # 残差连接
        out = self.relu(out)
        return out


class ResNet18(nn.Module):
    """适配 CIFAR-10 的 ResNet-18：首层用 3x3 卷积，不用 maxpool"""

    def __init__(self, num_classes=10):
        super().__init__()
        self.in_channels = 64
        # 首层：3x3 卷积，不降采样（保留 32x32）
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # 四个 stage，每个 stage 两个 BasicBlock
        self.layer1 = self._make_layer(64,  2, stride=1)   # 32x32
        self.layer2 = self._make_layer(128, 2, stride=2)   # 16x16
        self.layer3 = self._make_layer(256, 2, stride=2)   # 8x8
        self.layer4 = self._make_layer(512, 2, stride=2)   # 4x4

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_channels, out_channels, s))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

model = ResNet18(num_classes=10).to(device)

#====================损失，优化，调度====================
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(),lr=learning_rate,betas=(0.9,0.999),eps=1e-8,weight_decay=weight_decay)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch)

#====================训练====================
def train_epoch(epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs,labels in train_dataloader:
        inputs,labels = inputs.to(device),labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _,predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss/total
    train_acc = 100.0*correct/total
    return epoch_loss,train_acc

def evaluate(loader):
    model.eval()
    correct = 0
    total = 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / total

if __name__ == "__main__":
    print(device)
    best_acc = 0.0
    for i in range(1,epoch+1):
        loss,train_acc = train_epoch(i)
        test_acc = evaluate(test_dataloader)
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {i:2d}/{epoch} | LR {current_lr:.5f} | "
              f"Loss {loss:.4f} | Train {train_acc:.2f}% | Test {test_acc:.2f}%")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), "cifar10_resnet18_best.pth")
            print(f"  ✓ 已保存最佳模型 ({best_acc:.2f}%)")

    print(f"\n训练完成！最佳测试准确率: {best_acc:.2f}%")