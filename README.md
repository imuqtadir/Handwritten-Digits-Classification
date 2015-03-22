Note: The dataset has been taken from MNIST which contains around 50,000 handwritten samples of digits.


The source code contains following:

1. preprocess() : performs processing task and includes things such a dimensionality reduction. We divide our dataset into three parts, (i)Train Data (ii) Validation Data (iii) Test Data. We calculated the accuracy of our model once the training is done using the validation and the test datasets. 


2. sigmoid() : a simple sigmoid function where the input can be a scalar, vector or a matrix
 

3. nnObjFunction(): this contains the majority of the code for training the feed forward and error back propagation of our model. We compute the error function of the Neural Network.
 

4. nnPredict(): This function is used to predict the true lable for train data, validation data or test data.
 

5. initializeWeights(): this is used to randomly assign weights to the hidden units along with one bias unit.
 


The report talks about different sets of variations related to two major parameters,
	1. Accuracy VS Number of Hidden Units in Hidden Layer
	2. Accuracy VS Lambda 
	
	
Reference:


1. LeCun, Yann; Corinna Cortes, Christopher J.C. Burges. "MNIST handwritten digit database"
 

2. Bishop, Christopher M. "Pattern recognition and machine learning" (2007)
