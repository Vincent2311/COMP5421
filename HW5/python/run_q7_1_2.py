import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

batch_size = 120
max_iters = 30
learning_rate = 0.005
hidden_size = 64


train_dataloader = DataLoader(torchvision.datasets.MNIST(root='../data/MINSTdata', train=True,
                                      download=True, transform=transforms.ToTensor()),batch_size=batch_size,
                                           shuffle=True, num_workers=2)

test_dataloader = DataLoader(torchvision.datasets.MNIST(root='../data/MINSTdata', train=False,
                                      download=True, transform=transforms.ToTensor()),batch_size=batch_size,
                                           shuffle=False, num_workers=2)


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(2, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7*7*4, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x
    
model = ConvNet()

train_loss, train_accuracy = [],[]

criterion = nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

for itr in range(max_iters):
    train_total_loss = 0
    train_acc = 0  
    for images,labels in train_dataloader:
        # Forward pass

        outputs = model(images)
        predicted = torch.max(outputs, 1)[1]
        train_acc += predicted.eq(labels.data).sum().item() /labels.size()[0]
        loss = criterion(outputs, labels)
        train_total_loss += loss.detach().numpy()
        
        # Backward and optimize
        optim.zero_grad()
        loss.backward()
        optim.step()
        
    avg_accuracy = train_acc / len(train_dataloader)
    train_total_loss /= (len(train_dataloader) * batch_size)
    train_accuracy.append(avg_accuracy)
    train_loss.append(train_total_loss)

    if (itr+1) % 2 == 0:
            print ('Epoch [{}/{}], Acc: {:.4f}, Loss: {:.4f}' 
                   .format(itr+1, max_iters , avg_accuracy,loss.item()))
    
model.eval()
test_total_loss = 0
test_acc = 0
# Forward pass
for images,labels in test_dataloader:
    outputs = model(images)
    predicted = torch.max(outputs.data, 1)[1]
    test_acc += predicted.eq(labels.data).sum().item() /labels.size()[0]
    loss = criterion(outputs, labels)
    test_total_loss += loss

print("Test accuracy: ", test_acc/len(test_dataloader))
print("Test loss: ", test_total_loss/(len(test_dataloader)*batch_size))


plt.figure('accuracy')
plt.plot(range(max_iters), train_accuracy, color='b')
plt.legend(['training'])
plt.show()

plt.figure('loss')
plt.plot(range(max_iters), train_loss, color='g')
plt.legend(['training'])
plt.show()