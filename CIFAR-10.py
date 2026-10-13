import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
from torchvision import datasets,transforms

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,32,kernel_size=3,padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32,64,kernel_size=3,padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2,2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.25)

    def forward(self,x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

transform = transforms.Compose([
    transforms.ToTensor(),                          # 转为 Tensor，像素值归一化到 [0, 1]
    transforms.Normalize((0.1307,), (0.3081,))     # 标准化：减均值除标准差
])
train_dataset = datasets.CIFAR10(root="./data",train=True,download=True,transform=transform)
test_dataset = datasets.CIFAR10(root="./data",train=False,download=True,transform=transform)
train_loader = DataLoader(train_dataset,batch_size=100,shuffle=True)
test_loader = DataLoader(test_dataset,batch_size=1000,shuffle=False)

model = MyModel()
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model = model.to(device)

best_acc = 0.0

for epoch in range(30):
    model.train()
    for inputs,labels in train_loader:
        inputs,labels = inputs.to(device),labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs,labels in test_loader:
            inputs,labels = inputs.to(device),labels.to(device)
            outputs = model(inputs)
            _,predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    acc = 100.*correct/total
    print(f'Epoch {epoch},Loss {loss.item():.4f},Test Acc: {acc:.2f}%')

    scheduler.step()
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "cifar10_stage1_best.pth")