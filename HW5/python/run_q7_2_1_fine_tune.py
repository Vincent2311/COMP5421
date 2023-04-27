import torch
import torch.nn as nn
import torchvision
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision import models
from torch.autograd import Variable
import matplotlib.pyplot as plt

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def main():
  batch_size = 32
  num_of_epoch = 10
  learning_rate = 0.001
  train_transform = T.Compose([
    T.Resize(256),
    T. CenterCrop((224,224)),
    T.RandomHorizontalFlip(),
    T.ToTensor(),            
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
  ])

  train_dset = ImageFolder('../data/oxford-flowers17/train', transform=train_transform)
  train_loader = DataLoader(train_dset,
                      batch_size=batch_size,
                      num_workers=2,
                      shuffle=True)

  val_transform = T.Compose([
      T.Resize(224),
      T. CenterCrop((224,224)),
      T.ToTensor(),
      T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
  val_dset = ImageFolder('../data/oxford-flowers17/val', transform=val_transform)
  val_loader = DataLoader(val_dset,
                    batch_size=batch_size,
                    num_workers=2)
  
  test_transform = T.Compose([
      T.Resize(224),
      T. CenterCrop((224,224)),
      T.ToTensor(),
      T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
  test_dset = ImageFolder('../data/oxford-flowers17/test', transform=test_transform)
  test_loader = DataLoader(test_dset,
                    batch_size=batch_size,
                    num_workers=2)
  

  model = models.squeezenet1_1(pretrained=True)

  num_classes = len(train_dset.classes)
  model.classifier[1] = nn.Conv2d(512, num_classes, 1)
  model.num_classes = num_classes

  model.type(torch.FloatTensor)
  criterion = nn.CrossEntropyLoss().type(torch.FloatTensor)

  for param in model.parameters():
      param.requires_grad = False
  for param in model.classifier.parameters():
      param.requires_grad = True

  optimizer = torch.optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
  for epoch in range(num_of_epoch):
    # Run an epoch over the training data.
    run_epoch(model, criterion, train_loader, optimizer)

    # Check accuracy on the train and val sets.
    train_acc,train_loss = check_accuracy_loss(model, train_loader,criterion,batch_size)
    val_acc,val_loss = check_accuracy_loss(model, val_loader,criterion,batch_size)
    if (epoch+1) % 2 == 0:
            print ('P1Epoch [{}/{}], train_acc: {:.6f}, train_Loss: {:.6f}, val_acc: {:.6f}, val_loss: {:.6f}' 
                   .format(epoch+1, num_of_epoch ,train_acc,train_loss,val_acc,val_loss))

  # Now we want to finetune the entire model for a few epochs. To do thise we
  # will need to compute gradients with respect to all model parameters, so
  # we flag all parameters as requiring gradients.
  for param in model.parameters():
    param.requires_grad = True

  # Construct a new Optimizer that will update all model parameters. Note the
  # small learning rate.
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate/100)

  # Train the entire model for a few more epochs, checking accuracy on the
  # train and validation sets after each epoch.
  training_loss, training_accuracy = [],[]
  validation_loss, validation_accuracy = [],[]
  for epoch in range(num_of_epoch):
    run_epoch(model, criterion, train_loader, optimizer)

    train_acc, train_loss = check_accuracy_loss(model, train_loader,criterion,batch_size)
    val_acc,val_loss = check_accuracy_loss(model, val_loader,criterion,batch_size)
    training_loss.append(train_loss)
    validation_loss.append(val_loss)
    training_accuracy.append(train_acc)
    validation_accuracy.append(val_acc)
    if (epoch+1) % 2 == 0:
      print ('P2Epoch [{}/{}], train_acc: {:.6f}, train_Loss: {:.6f}, val_acc: {:.6f}, val_loss: {:.6f}' 
                   .format(epoch+1, num_of_epoch ,train_acc,train_loss,val_acc,val_loss))
  
  plt.figure('accuracy')
  plt.plot(range(num_of_epoch), training_accuracy, color='b')
  plt.plot(range(num_of_epoch), validation_accuracy, color='g')
  plt.legend(['training','validation'])
  plt.show()
  plt.savefig('accuracy.png')

  plt.figure('loss')
  plt.plot(range(num_of_epoch), training_loss, color='b')
  plt.plot(range(num_of_epoch), validation_loss, color='g')
  plt.legend(['training','validation'])
  plt.show()
  plt.savefig('loss.png')

  # Test period
  test_acc,test_loss = check_accuracy_loss(model, test_loader,criterion,batch_size)
  print('Test loss: ', test_loss)
  print('Test accuracy: ', test_acc)


def run_epoch(model, loss_fn, loader, optimizer):
  """
  Train the model for one epoch.
  """
  # Set the model to training mode
  model.train()
  for x, y in loader:
    # The DataLoader produces Torch Tensors, so we need to cast them to the
    # correct datatype and wrap them in Variables.
    #
    # Note that the labels should be a torch.LongTensor on CPU and a
    # torch.cuda.LongTensor on GPU; to accomplish this we first cast to dtype
    # (either torch.FloatTensor or torch.cuda.FloatTensor) and then cast to
    # long; this ensures that y has the correct type in both cases.
    x_var = Variable(x.type(torch.FloatTensor))
    y_var = Variable(y.type(torch.FloatTensor).long())

    # Run the model forward to compute scores and loss.
    scores = model(x_var)
    loss = loss_fn(scores, y_var)

    # Run the model backward and take a step using the optimizer.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def check_accuracy_loss(model, loader,criterion,batch_size):
  """
  Check the accuracy of the model.
  """
  # Set the model to eval mode
  model.eval()
  num_correct, num_samples,total_loss = 0, 0, 0
  for x, y in loader:
    # Cast the image data to the correct type and wrap it in a Variable. At
    # test-time when we do not need to compute gradients, marking the Variable
    # as volatile can reduce memory usage and slightly improve speed.
    x_var = Variable(x.type(torch.FloatTensor))
    
    # Run the model forward, and compare the argmax score with the ground-truth
    # category.
    scores = model(x_var)
    _, preds = scores.data.cpu().max(1)
    total_loss += criterion(scores, y).detach().numpy()
    num_correct += (preds == y).sum()
    num_samples += x.size(0)

  # Return the fraction of datapoints that were correctly classified.
  acc = float(num_correct) / num_samples
  loss = float(total_loss) / num_samples
  return acc,loss


if __name__ == '__main__':
  main()