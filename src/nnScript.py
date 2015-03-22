# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import pickle
import time

###################

def initializeWeights(n_in,n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
       
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""
    
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1);
    W = (np.random.rand(n_out, n_in + 1)*2* epsilon) - epsilon;
    return W

#########################
def sigmoid(z):
    
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""
    return np.reciprocal((1+np.exp(-1*z) ))


########################

def preprocess():
    mat = loadmat('F:/sem2masters/ML/basecode/basecode/mnist_all.mat') 

    zeroArray =[1,0,0,0,0,0,0,0,0,0]
    oneArray = [0,1,0,0,0,0,0,0,0,0]
    twoArray = [0,0,1,0,0,0,0,0,0,0]
    threeArray = [0,0,0,1,0,0,0,0,0,0]
    fourArray=[0,0,0,0,1,0,0,0,0,0]
    fiveArray = [0,0,0,0,0,1,0,0,0,0]
    sixArray = [0,0,0,0,0,0,1,0,0,0]
    sevenArray = [0,0,0,0,0,0,0,1,0,0]
    eightArray = [0,0,0,0,0,0,0,0,1,0]
    nineArray = [0,0,0,0,0,0,0,0,0,1]


    train0 = mat.get('train0')
    train1 = mat.get('train1')
    #print train1.shape
    train2 = mat.get('train2')
    #print train2.shape
    train3 = mat.get('train3')
    #print train3.shape
    train4 = mat.get('train4')
    #print train4.shape
    train5 = mat.get('train5')
    #print train5.shape
    train6 = mat.get('train6')
    #print train6.shape
    train7 = mat.get('train7')
    #print train7.shape
    train8 = mat.get('train8')
    #print train8.shape
    train9 = mat.get('train9')
    #print train9.shape

    #Forming of test_label
    test_label = [1,0,0,0,0,0,0,0,0,0]
    test0 = mat.get('test0')
    for j in range(test0.shape[0]-1):
        test_label = np.vstack([test_label, zeroArray])

    test1 = mat.get('test1')
    for j in range(test1.shape[0]):
         test_label = np.vstack([test_label, oneArray])


    test2 = mat.get('test2')
    for j in range(test2.shape[0]):
         test_label = np.vstack([test_label, twoArray])


    test3 = mat.get('test3')
    for j in range(test3.shape[0]):
         test_label = np.vstack([test_label, threeArray])
    
    test4 = mat.get('test4')
    for j in range(test4.shape[0]):
         test_label = np.vstack([test_label, fourArray])
    
    test5 = mat.get('test5')
    for j in range(test5.shape[0]):
         test_label = np.vstack([test_label, fiveArray])
    
    test6 = mat.get('test6')
    for j in range(test6.shape[0]):
         test_label = np.vstack([test_label, sixArray])
    
    test7 = mat.get('test7')
    for j in range(test7.shape[0]):
         test_label = np.vstack([test_label, sevenArray])
    
    test8 = mat.get('test8')
    for j in range(test8.shape[0]):
         test_label = np.vstack([test_label, eightArray])
    
    test9 = mat.get('test9')
    for j in range(test9.shape[0]):
         test_label = np.vstack([test_label, nineArray])

    test_data1 = np.vstack((test0,test1,test2,test3,test4,test5,test6,test7,test8,test9))

    train_label = [1,0,0,0,0,0,0,0,0,0]
    
    validation_label = [1,0,0,0,0,0,0,0,0,0]
    a = range(train0.shape[0])
    aperm = np.random.permutation(a)
    A1 = train0[aperm[0:1000],:]      
    A2 = train0[aperm[1000:],:]

    for j in range(A1.shape[0]-1):
        validation_label = np.vstack([validation_label, zeroArray])
    for j in range(A2.shape[0]-1):
        
        train_label = np.vstack([train_label, zeroArray])


    b = range(train1.shape[0])
    bperm = np.random.permutation(b)
    B1 = train1[bperm[0:1000],:]
    B2 = train1[bperm[1000:],:]

    for j in range(B1.shape[0]):
        validation_label = np.vstack([validation_label, oneArray])
    for i in range(B2.shape[0]):
        train_label = np.vstack([train_label, oneArray])


    c = range(train2.shape[0])
    cperm = np.random.permutation(c)
    C1 = train2[cperm[0:1000],:]
    C2 = train2[cperm[1000:],:]
    for j in range(C1.shape[0]):
        validation_label = np.vstack([validation_label, twoArray])
    for i in range(C2.shape[0]):
        train_label = np.vstack([train_label, twoArray])


    d = range(train3.shape[0])
    dperm = np.random.permutation(d)
    D1 = train3[dperm[0:1000],:]
    D2 = train3[dperm[1000:],:]
    for j in range(D1.shape[0]):
        validation_label = np.vstack([validation_label, threeArray])
    for i in range(D2.shape[0]):
        train_label = np.vstack([train_label, threeArray])


    e = range(train4.shape[0])
    eperm = np.random.permutation(e)
    E1 = train4[eperm[0:1000],:]
    E2 = train4[eperm[1000:],:]
    for j in range(E1.shape[0]):
        validation_label = np.vstack([validation_label, fourArray])
    for i in range(E2.shape[0]):
        train_label = np.vstack([train_label, fourArray])


    f = range(train5.shape[0])
    fperm = np.random.permutation(f)
    F1 = train5[fperm[0:1000],:]
    F2 = train5[fperm[1000:],:]
    for j in range(F1.shape[0]):
        validation_label = np.vstack([validation_label, fiveArray])
    for i in range(F2.shape[0]):
        train_label = np.vstack([train_label, fiveArray])


    g = range(train6.shape[0])
    gperm = np.random.permutation(g)
    G1 = train6[gperm[0:1000],:]
    G2 = train6[gperm[1000:],:]
    for j in range(G1.shape[0]):
        validation_label = np.vstack([validation_label, sixArray])
    for i in range(G2.shape[0]):
        train_label = np.vstack([train_label, sixArray])

    h = range(train7.shape[0])
    hperm = np.random.permutation(h)
    H1 = train7[hperm[0:1000],:]
    H2 = train7[hperm[1000:],:]
    for j in range(H1.shape[0]):
        validation_label = np.vstack([validation_label, sevenArray])
    for i in range(H2.shape[0]):
        train_label = np.vstack([train_label, sevenArray])



    i = range(train8.shape[0])
    iperm = np.random.permutation(i)
    I1 = train8[iperm[0:1000],:]
    I2 = train8[iperm[1000:],:]
    for j in range(I1.shape[0]):
        validation_label = np.vstack([validation_label, eightArray])
    for i in range(I2.shape[0]):
        train_label = np.vstack([train_label, eightArray])



    j = range(train9.shape[0])
    jperm = np.random.permutation(j)
    J1 = train9[jperm[0:1000],:]
    J2 = train9[jperm[1000:],:]
    for j in range(J1.shape[0]):
        validation_label = np.vstack([validation_label, nineArray])
    for i in range(J2.shape[0]):
        train_label = np.vstack([train_label, nineArray])

    validation_data1=np.vstack((A1,B1,C1,D1,E1,F1,G1,H1,I1,J1))
    train_data1=np.vstack((A2,B2,C2,D2,E2,F2,G2,H2,I2,J2))
    train_data1=np.double(train_data1)
    newarr1=np.ndarray(shape=(train_data1.shape[0],0))
    newarr2=np.ndarray(shape=(validation_data1.shape[0],0))
    newarr3=np.ndarray(shape=(test_data1.shape[0],0))
    for i in (ind for ind, i in np.ndenumerate(np.std(train_data1, axis=0)) if i > 0.1):
        a= train_data1[:, i]
        b = validation_data1[:,i]
        c = test_data1[:,i]
        newarr1=np.hstack((newarr1,a))
        newarr2=np.hstack((newarr2,b))
        newarr3=np.hstack((newarr3,c))
    
    train_data = newarr1
    validation_data = newarr2
    test_data = newarr3   
    validation_data = np.divide(validation_data,255.0)
    test_data = np.divide(test_data,255.0)
    train_data = np.divide(train_data,255.0)
    return train_data, train_label, validation_data, validation_label, test_data, test_label

####################################

def nnObjFunction(params, *args):

    n_input, n_hidden, n_class, train_data, train_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    
    newarr1=np.ones((train_data.shape[0],1))
    train_data=np.hstack((train_data,newarr1))
    obj_val = 0  

    A=np.dot(train_data,w1.T)
    A = np.double(A)

    Z=sigmoid(A)
    
    Z = np.double(Z)
    newarr1=np.ones((train_data.shape[0],1))
    Z=np.hstack((Z,newarr1))

    B=np.dot(Z,w2.T)
    O=sigmoid(B)
 
    
    #Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    #you would use code similar to the one below to create a flat array
   
    grad_w1 = np.empty(w1.shape)
    grad_w1.fill(0.0)
    grad_w2 = np.empty(w2.shape)
    grad_w2.fill(0.0)

    delta=O-train_label
    DeltaProductWj=np.dot(delta,w2)
    prefix = DeltaProductWj*Z*(1-Z)
    grad_w1=np.dot(prefix.T, train_data)
    grad_w1 = grad_w1[:-1,:]
    grad_w2 = np.dot(delta.T,Z)

    grad_w1=(grad_w1+lambdaval*w1)/train_data.shape[0];
    grad_w2=(grad_w2+lambdaval*w2)/train_data.shape[0];

    obj_grad = np.array([])
    obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)

    J1=(train_label*np.log(O))+((1-train_label)*(np.log((1-O))))

    summing = obj_val + np.sum(J1)

    inter_obj_val = -1*(summing/train_data.shape[0])
    lambdabyTwoN = lambdaval/(2*train_data.shape[0])
    obj_val = inter_obj_val+ (lambdabyTwoN *(np.sum(w1*w1) + np.sum(w2*w2)))
    #print 'obj_val is '
    #print obj_val
    return (obj_val,obj_grad)

########################


def nnPredict(w1,w2,data):
    
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature 
    %       vector of a particular image
       
    % Output: 
    % label: a column vector of predicted labels""" 

    labels = np.array([])
    flag = False
    
    Z = np.zeros((1,w1.shape[0]+1))
    Z[0][w1.shape[0]] = 1
    for image in range(data.shape[0]): 
        current_row = np.concatenate((data[image,:],[1])) 
        current_row = current_row.reshape(1,current_row.size)
        Z[0][:-1] = np.dot(current_row,w1.T)
        Z[0][:-1] = sigmoid(Z[0][:-1])
        O = np.dot(Z,w2.T)
        O = sigmoid(O)
        Label_val = np.array([0]*10)
        Label_val[np.argmax(O)] = 1
        temp = np.argmax(Label_val)
       
        if (flag == True):
            labels = np.vstack((labels,temp))
        
        else:
            flag = True
            labels = np.copy(temp)

    return labels

############


start_time = time.time()
train_data, train_label, validation_data,validation_label, test_data, test_label = preprocess();
#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]; 

# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 70;
				   
# set the number of nodes in output unit
n_class = 10;				   

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden);
initial_w2 = initializeWeights(n_hidden, n_class);

# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()),0)

# set the regularization hyper-parameter
lambdaval = 0.5;


args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

#Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

opts = {'maxiter' : 70}    # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args,method='CG', options=opts)

#In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
#and nnObjGradient. Check documentation for this function before you proceed.
#nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


#Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

with open('params.pkl', 'wb') as output:
    pickle.dump(w1, output, -1)
    pickle.dump(w2, output,-1)
    pickle.dump(n_hidden, output,-1)
    pickle.dump(lambdaval,output,-1)


#Test the computed parameters

predicted_label = nnPredict(w1,w2,train_data)
flag = False

train_label1=np.array([])
for row in range(train_label.shape[0]): 
    temp = np.argmax(train_label[row,:])
    
    if (flag == True):
        train_label1 = np.vstack((train_label1,temp))
    
    else:
        flag = True
        train_label1 = np.copy(temp)
        


print('\n Training set Accuracy:' + str(100*np.mean((predicted_label == train_label1).astype(float))) + '%')

predicted_label = nnPredict(w1,w2,validation_data)
flag = False
validation_label1=np.array([])

for row in range(validation_label.shape[0]): 
    temp = np.argmax(validation_label[row,:])

    if (flag == True):
        validation_label1 = np.vstack((validation_label1,temp))
    else:
        flag = True
        validation_label1 = np.copy(temp)


print('\n Validation set Accuracy:' + str(100*np.mean((predicted_label == validation_label1).astype(float))) + '%')


predicted_label = nnPredict(w1,w2,test_data)
flag = False

test_label1=np.array([])
for row in range(test_label.shape[0]): 
    temp = np.argmax(test_label[row,:])
    
    if (flag == True):
        test_label1 = np.vstack((test_label1,temp))
   
    else:
        flag = True
        test_label1 = np.copy(temp)

print('\n Test set Accuracy:' + str(100*np.mean((predicted_label == test_label1).astype(float))) + '%')

print('Time taken %s seconds'% (time.time() - start_time))
