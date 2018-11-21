import numpy as np
import matplotlib.pyplot as plt

def init_two_layer_model(input_size, hidden_size, output_size):
  """
  Initialize the weights and biases for a two-layer fully connected neural
  network. The net has an input dimension of D, a hidden layer dimension of H,
  and performs classification over C classes. Weights are initialized to small
  random values and biases are initialized to zero.

  Inputs:
  - input_size: The dimension D of the input data
  - hidden_size: The number of neurons H in the hidden layer
  - ouput_size: The number of classes C

  Returns:
  A dictionary mapping parameter names to arrays of parameter values. It has
  the following keys:
  - W1: First layer weights; has shape (D, H)
  - b1: First layer biases; has shape (H,)
  - W2: Second layer weights; has shape (H, C)
  - b2: Second layer biases; has shape (C,)
  """
  # initialize a model
  model = {}
  model['W1'] = 0.00001 * np.random.randn(input_size, hidden_size)
  model['b1'] = np.zeros(hidden_size)
  model['W2'] = 0.00001 * np.random.randn(hidden_size, output_size)
  model['b2'] = np.zeros(output_size)
  return model

def two_layer_net(X, model, y=None, reg=0.0):
  """
  Compute the loss and gradients for a two layer fully connected neural network.
  The net has an input dimension of D, a hidden layer dimension of H, and
  performs classification over C classes. We use a softmax loss function and L2
  regularization the the weight matrices. The two layer net should use a ReLU
  nonlinearity after the first affine layer.

  The two layer net has the following architecture:

  input - fully connected layer - ReLU - fully connected layer - softmax

  The outputs of the second fully-connected layer are the scores for each
  class.

  Inputs:
  - X: Input data of shape (N, D). Each X[i] is a training sample.
  - model: Dictionary mapping parameter names to arrays of parameter values.
    It should contain the following:
    - W1: First layer weights; has shape (D, H)
    - b1: First layer biases; has shape (H,)
    - W2: Second layer weights; has shape (H, C)
    - b2: Second layer biases; has shape (C,)
  - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
    an integer in the range 0 <= y[i] < C. This parameter is optional; if it
    is not passed then we only return scores, and if it is passed then we
    instead return the loss and gradients.
  - reg: Regularization strength.

  Returns:
  If y not is passed, return a matrix scores of shape (N, C) where scores[i, c]
  is the score for class c on input X[i].

  If y is not passed, instead return a tuple of:
  - loss: Loss (data loss and regularization loss) for this batch of training
    samples.
  - grads: Dictionary mapping parameter names to gradients of those parameters
    with respect to the loss function. This should have the same keys as model.
  """

  # unpack variables from the model dictionary
  W1,b1,W2,b2 = model['W1'], model['b1'], model['W2'], model['b2']
  N, D = X.shape
  eps = 1e-15
  # compute the forward pass
  scores = None
  #############################################################################
  # TODO: Perform the forward pass, computing the class scores for the input. #
  # Store the result in the scores variable, which should be an array of      #
  # shape (N, C).                                                             #
  #############################################################################
  a0 = X
  z1 = np.dot(a0, W1) + b1.reshape(1,-1)
  a1 = np.maximum(0, z1)
  z2 = np.dot(a1, W2) + b2.reshape(1,-1)
  scores = z2
  #############################################################################
  #                              END OF YOUR CODE                             #
  #############################################################################
  
  # If the targets are not given then jump out, we're done
  if y is None:
    return scores

  # compute the loss
  loss = None
  #############################################################################
  # TODO: Finish the forward pass, and compute the loss. This should include  #
  # both the data loss and L2 regularization for W1 and W2. Store the result  #
  # in the variable loss, which should be a scalar. Use the Softmax           #
  # classifier loss. So that your results match ours, multiply the            #
  # regularization loss by 0.5                                                #
  #############################################################################

  def softmax(x):
    x -= np.max(x)
    return np.exp(x)/np.sum(np.exp(x))

  def softmax_prime(s):
  	s[range(N), y] -= 1
  	s /= N
  	return s

  def relu_prime(da, Z):
  	dz = np.copy(da)
  	dz[Z < 0] = 0
  	return dz

  def back_prop(dz, cache):
  	A_prev, W, b = cache
  	dW = np.dot(A_prev.T, dz)
  	db = np.sum(dz, axis=0, keepdims=True)
  	dA_prev = np.dot(dz, W.T)
  	dW += reg * W
  	return dA_prev, dW, db

  a2 = np.zeros(scores.shape)

  for i in range(len(scores)):
  	a2[i] = softmax(scores[i])

  data_loss = -np.log(a2 + eps)[np.arange(N), y]
  l2_reg = 0.5 * (np.sum(np.power(W1,2)) + np.sum(np.power(W2,2)))
  loss = np.sum(data_loss)/N + (reg * l2_reg)

  #############################################################################
  #                              END OF YOUR CODE                             #
  #############################################################################

  #############################################################################
  # TODO: Compute the backward pass, computing the derivatives of the weights #
  # and biases. Store the results in the grads dictionary. For example,       #
  # grads['W1'] should store the gradient on W1, and be a matrix of same size #
  #############################################################################
  dz2 = softmax_prime(a2)

  da1, dW2, db2 = back_prop(dz2, [a1, W2, b2])
  dz1 = relu_prime(da1, z1)
  da0, dW1, db1 = back_prop(dz1, [a0, W1, b1])
  db1 = db1.reshape(-1)
  db2 = db2.reshape(-1)
  # compute the gradients
  grads = {'W1': dW1, 'W2': dW2, 'b1': db1, 'b2': db2}
  #############################################################################
  #                              END OF YOUR CODE                             #
  #############################################################################

  return loss, grads


















