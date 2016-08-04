import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  N = X.shape[0]
  D = X.shape[1]

  score = np.dot(X,W)
  score -= np.max(score)
  probs = np.exp(score) / np.sum(np.exp(score), axis=1, keepdims=True)

  # Cross Entropy loss
  loss = -np.sum(np.log(probs[range(N),y]))/N + 0.5*reg*np.sum(W*W)
  loss = float(loss)

  # CE grad. Using my Deep for NLP results: X[i]*(probs - ind(i==y[i]))
  dW = np.dot(X.T,probs)
  dW[range(N),y] -= 1

  dW = dW/N + reg*W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  N = X.shape[0]
  D = X.shape[1]

  score = np.dot(X,W)
  score -= np.max(score)
  probs = np.exp(score) / np.sum(np.exp(score), axis=1, keepdims=True)

  # Cross Entropy loss
  loss = -np.sum(np.log(probs[range(N),y]))/N
  loss = float(loss)

  # CE grad. Using my Deep for NLP results: X[i]*(probs - ind(i==y[i]))
  dW = np.dot(X.T,probs)
  dW[range(N),y] -= 1

  dW = dW/N

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

