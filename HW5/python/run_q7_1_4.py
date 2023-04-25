import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.patches
import skimage
import string
from q4 import findLetters

batch_size = 120
max_iters = 30
learning_rate = 0.005
hidden_size = 64


train_dataloader = DataLoader(torchvision.datasets.EMNIST(root='../data/EMINSTdata',split = 'balanced', train=True,
                                      download=True, transform=transforms.ToTensor()),batch_size=batch_size,
                                           shuffle=True, num_workers=2)

test_dataloader = DataLoader(torchvision.datasets.EMNIST(root='../data/EMINSTdata', split = 'balanced',train=False,
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
        self.fc = nn.Linear(7*7*4, 47)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x
    
# model = ConvNet()

# train_loss, train_accuracy = [],[]

# criterion = nn.CrossEntropyLoss()
# optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

# for itr in range(max_iters):
#     train_total_loss = 0
#     train_acc = 0  
#     for images,labels in train_dataloader:
#         # Forward pass

#         outputs = model(images)
#         predicted = torch.max(outputs, 1)[1]
#         train_acc += predicted.eq(labels.data).sum().item() /labels.size()[0]
#         loss = criterion(outputs, labels)
#         train_total_loss += loss.detach().numpy()
        
#         # Backward and optimize
#         optim.zero_grad()
#         loss.backward()
#         optim.step()
        
#     avg_accuracy = train_acc / len(train_dataloader)
#     train_total_loss /= (len(train_dataloader) * batch_size)
#     train_accuracy.append(avg_accuracy)
#     train_loss.append(train_total_loss)

#     if (itr+1) % 2 == 0:
#             print ('Epoch [{}/{}], Acc: {:.4f}, Loss: {:.4f}' 
#                    .format(itr+1, max_iters , avg_accuracy,loss.item()))

# plt.figure('accuracy')
# plt.plot(range(max_iters), train_accuracy, color='b')
# plt.legend(['training'])
# plt.show()

# plt.figure('loss')
# plt.plot(range(max_iters), train_loss, color='g')
# plt.legend(['training'])
# plt.show()

# torch.save(model.state_dict(),'run_q7_1_4.pth')

model = ConvNet()
model.load_state_dict(torch.load('run_q7_1_4.pth'))

def get_rata_row(img):
    bboxes, bw = findLetters(img)

    plt.imshow(bw, cmap='gray')
    for bbox in bboxes:
        minr, minc, maxr, maxc = bbox
        rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                fill=False, edgecolor='red', linewidth=2)
        plt.gca().add_patch(rect)
    plt.show()

    # find the rows using..RANSAC, counting, clustering, etc.
    mean_height = sum([b[2]-b[0] for b in bboxes])/len(bboxes)
    #get the center_x, center_y, width, height 
    centers = [((b[3]+b[1])//2,(b[2]+b[0])//2,b[3]-b[1],b[2]-b[0]) for b in bboxes]
    centers = sorted(centers,key = lambda center: center[1])

    rows = []
    row = []
    current_y = centers[0][1]
    for center in centers:
        if center[1] - current_y > mean_height:
            row = sorted(row,key = lambda center: center[0])
            rows.append(row)
            row = [center]
            current_y = center[1]
        else:
            row.append(center)
    row = sorted(row,key = lambda center: center[0])
    rows.append(row)

    # crop the bounding boxes
    # note.. before you flatten, transpose the image (that's how the dataset is!)
    # consider doing a square crop, and even using np.pad() to get your images looking more like the dataset
    data = []
    for row in rows:
        line = []
        for x,y,width,height in row:
            cropped = bw[y-height//2:y+height//2, x-width//2:x+width//2]
            if height > width:
                padding=((10,10),((height-width)//2 + 10,(height-width)//2 + 10))
            else:
                padding=(((width-height)//2 + width//10,(width-height)//2 + 10),(10,10))
            cropped = np.pad(cropped,padding,mode='constant',constant_values=(1, 1))
            cropped = skimage.transform.resize(cropped, (28, 28))
            cropped = skimage.morphology.erosion(cropped)
            line.append(1-cropped.T)
        data.append(np.array(line))
    return data



model.eval()
# test_total_loss = 0
# test_acc = 0
# # Forward pass
# criterion = nn.CrossEntropyLoss()
# for images,labels in test_dataloader:
#     outputs = model(images)
#     predicted = torch.max(outputs.data, 1)[1]
#     test_acc += predicted.eq(labels.data).sum().item() /labels.size()[0]
#     loss = criterion(outputs, labels)
#     test_total_loss += loss

# print("Test accuracy: ", test_acc/len(test_dataloader))
# print("Test loss: ", test_total_loss/(len(test_dataloader)*batch_size))
    
letters = np.array([str(_) for _ in range(10)] + [_ for _ in string.ascii_uppercase[:26]]
            + ['a'] + ['b'] + ['d'] + ['e'] + ['f'] + ['g'] + ['h'] + ['n'] + ['q'] + ['r'] + ['t'])
# Forward pass
for img in os.listdir('../images'):
    img = skimage.img_as_float(skimage.io.imread(os.path.join('../images',img)))
    data = get_rata_row(img)
    
    for line in data:
        line = torch.from_numpy(line).type(torch.float32).unsqueeze(1)
        outputs = model(line)
        pred_idx = torch.max(outputs, 1)[1].numpy()
        strings = ''
        for idx in pred_idx:
            strings += letters[idx]
        print(strings)
    


