import torch
import scipy.io
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

batch_size = 120
max_iters = 80
learning_rate = 0.005
hidden_size = 64


train_data = scipy.io.loadmat('../data/nist36_train.mat')
test_data = scipy.io.loadmat('../data/nist36_test.mat')

train_x, train_y = torch.from_numpy(train_data['train_data']).type(torch.float32), torch.from_numpy(train_data['train_labels']).type(torch.float32)
test_x, test_y = torch.from_numpy(test_data['test_data']).type(torch.float32), torch.from_numpy(test_data['test_labels']).type(torch.float32)


train_dataloader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(TensorDataset(test_x, test_y), batch_size=batch_size, shuffle=True)

class ConvNet(nn.Module):

    def __init__(self):
        super(ConvNet, self).__init__()
        # an affine operation: y = Wx + b
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7*7*32, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x
    
model = ConvNet()

train_loss, train_accuracy = [],[]
validation_loss, validation_accuracy = [],[]
valid_acc, valid_loss = None, None

criterion = nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

for itr in range(max_iters):
    train_total_loss = 0
    train_acc = 0
    # model.train()   
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