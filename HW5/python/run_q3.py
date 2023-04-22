import numpy as np
import scipy.io
from nn import *
import matplotlib.pyplot as plt

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']

max_iters = 50
# pick a batch size, learning rate
batch_size = 32
learning_rate = 0.01
hidden_size = 64

batches = get_random_batches(train_x,train_y,batch_size)
batch_num = len(batches)

params = {}

# initialize layers here
initialize_weights(train_x.shape[1],hidden_size,params,'layer1')
initialize_weights(hidden_size,train_y.shape[1],params,'output')

import copy
Wlayer1_orig = copy.deepcopy(params['Wlayer1'])

train_loss, train_accuracy = [],[]
validation_loss, validation_accuracy = [],[]
valid_acc, valid_loss = None, None
# with default settings, you should get loss < 150 and accuracy > 80%
for itr in range(max_iters):
    total_loss = 0
    total_acc = 0
    for xb,yb in batches:
        # forward
        h1 = forward(xb,params,'layer1')
        probs = forward(h1,params,'output',softmax)
        # loss
        loss, acc = compute_loss_and_acc(yb,probs)
        # be sure to add loss and accuracy to epoch totals 
        total_loss += loss 
        total_acc += acc
        # backward
        delta1 = probs - yb
        delta2 = backwards(delta1,params,'output',linear_deriv)
        backwards(delta2,params,'layer1',sigmoid_deriv)
        # apply gradient
        params["Wlayer1"] -= learning_rate*params["grad_Wlayer1"]
        params["blayer1"] -= learning_rate*params["grad_blayer1"]
        params["Woutput"] -= learning_rate*params["grad_Woutput"]
        params["boutput"] -= learning_rate*params["grad_boutput"]

    total_acc /= batch_num
    total_loss/= (batch_num*batch_size)
    train_loss.append(total_loss)
    train_accuracy.append(total_acc)    
    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr,total_loss,total_acc))
    
    # run on validation set and report accuracy! should be above 75%
    
    # forward
    h1 = forward(valid_x,params,'layer1')
    probs = forward(h1,params,'output',softmax)
    # loss
    valid_loss, valid_acc = compute_loss_and_acc(valid_y,probs)
    valid_loss /= valid_x.shape[0]
    validation_loss.append(valid_loss)
    validation_accuracy.append(valid_acc) 

print('Validation accuracy: ',valid_acc)
print('Validation loss: ',valid_loss)

plt.figure('accuracy')
plt.plot(range(max_iters), train_accuracy, color='g')
plt.plot(range(max_iters), validation_accuracy, color='b')
plt.legend(['training', 'validation'])
plt.show()

plt.figure('loss')
plt.plot(range(max_iters), train_loss, color='g')
plt.plot(range(max_iters), validation_loss, color='b')
plt.legend(['training', 'validation'])
plt.show()

test_data = scipy.io.loadmat('../data/nist36_test.mat')
test_x, test_y = test_data['test_data'], test_data['test_labels']
h1 = forward(test_x,params,'layer1')
probs = forward(h1,params,'output',softmax)
test_loss, test_acc = compute_loss_and_acc(test_y,probs)
test_loss /= test_x.shape[0]
print("training_acc: ",test_acc)


if False: # view the data
    for crop in xb:
        import matplotlib.pyplot as plt
        plt.imshow(crop.reshape(32,32).T)
        plt.show()
import pickle
saved_params = {k:v for k,v in params.items() if '_' not in k}
with open('q3_weights.pickle', 'wb') as handle:
    pickle.dump(saved_params, handle, protocol=pickle.HIGHEST_PROTOCOL)


# Q3.1.3
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

fig = plt.figure()
grid = ImageGrid(fig, 111,  
                 nrows_ncols=(8, 8),  
                 axes_pad=0.01, 
                 )
for i in range(hidden_size):
    grid[i].imshow(np.reshape(Wlayer1_orig[:,i],(32,32)))
    plt.axis('off')
plt.show()

fig = plt.figure()
grid = ImageGrid(fig, 111,  
                 nrows_ncols=(8, 8),
                 axes_pad=0.01,  
                 )
for i in range(hidden_size):
    grid[i].imshow(np.reshape(params['Wlayer1'][:,i],(32,32)))
    plt.axis('off')
plt.show()

# Q3.1.3

def generate_confusion_matrix(probs, y):
    confusion_matrix = np.zeros((y.shape[1],y.shape[1]))
    predict_idx = np.argmax(probs,axis=1)
    actual_idx = np.argmax(y,axis=1)
    for i in range(y.shape[0]):
        confusion_matrix[predict_idx[i],actual_idx[i]] +=1 
    return confusion_matrix

import string

# training
h1 = forward(train_x,params,'layer1')
probs = forward(h1,params,'output',softmax)
test_confusion_matrix = generate_confusion_matrix(probs,train_y)
plt.imshow(test_confusion_matrix,interpolation='nearest')
plt.grid(True)
plt.xticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.yticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.show()


# validation
h1 = forward(valid_x,params,'layer1')
probs = forward(h1,params,'output',softmax)
valid_confusion_matrix = generate_confusion_matrix(probs,valid_y)
plt.imshow(valid_confusion_matrix,interpolation='nearest')
plt.grid(True)
plt.xticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.yticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.show()


# testing
h1 = forward(test_x,params,'layer1')
probs = forward(h1,params,'output',softmax)
test_confusion_matrix = generate_confusion_matrix(probs,test_y)
plt.imshow(test_confusion_matrix,interpolation='nearest')
plt.grid(True)
plt.xticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.yticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.show()