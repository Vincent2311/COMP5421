import torch
import scipy.io
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

batch_size = 120
max_iters = 40
learning_rate = 0.0025
hidden_size = 64

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')
test_data = scipy.io.loadmat('../data/nist36_test.mat')

train_x, train_y = torch.from_numpy(train_data['train_data']).type(torch.float32), torch.from_numpy(train_data['train_labels']).type(torch.float32)
valid_x, valid_y = torch.from_numpy(valid_data['valid_data']).type(torch.float32), torch.from_numpy(valid_data['valid_labels']).type(torch.float32)
test_x, test_y = torch.from_numpy(test_data['test_data']).type(torch.float32), torch.from_numpy(test_data['test_labels']).type(torch.float32)


train_dataloader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(TensorDataset(valid_x, valid_y), batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(TensorDataset(test_x, test_y), batch_size=batch_size, shuffle=True)


class Net(nn.Module):

    def __init__(self,D_in,Hidden, D_out):
        super(Net, self).__init__()
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(D_in, Hidden)  # 5*5 from image dimension
        self.fc2 = nn.Linear(Hidden, D_out)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = self.fc2(x)
        return x
    
model = Net(train_x.shape[1], hidden_size, train_y.shape[1])

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
        targets = torch.max(labels, 1)[1]
        outputs = model(images)
        predicted = torch.max(outputs, 1)[1]
        train_acc += predicted.eq(targets.data).sum().item() /labels.size()[0]
        loss = criterion(outputs, targets)
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
    
    # model.eval()
    # # Forward pass
    # valid_total_loss = 0
    # valid_acc = 0
    # for images,labels in valid_dataloader:
    #     outputs = model(images)
    #     predicted = torch.max(outputs.data, 1)[1]
    #     targets = torch.max(labels, 1)[1]
    #     valid_acc = predicted.eq(targets.data).sum().item() /labels.size()[0]
    #     loss = criterion(outputs, labels)
    #     valid_total_loss+=loss
    
    # validation_loss.append(valid_total_loss/(len(valid_dataloader)*batch_size))
    # validation_accuracy.append(valid_acc/len(valid_dataloader))


model.eval()
test_total_loss = 0
test_acc = 0
# Forward pass
for images,labels in test_dataloader:
    outputs = model(images)
    predicted = torch.max(outputs.data, 1)[1]
    targets = torch.max(labels, 1)[1]
    test_acc += predicted.eq(targets.data).sum().item() /labels.size()[0]
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