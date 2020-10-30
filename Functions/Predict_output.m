function [OUT] = Predict_output(X,weights)
%Perform Neural Network Prediction with forward propagation using the
%weigths after the training process. 
%==========================================================================
%Input: - X --> Training examples to predict their outcomes. [N x MxIP]
%               With N the total number of observations and MxIP rergessors 
%               (number of neurons in the input layer)
%
%         weights--> weights after the training [M x MxIP]
%
%Outputs: - OUT--> Predicted outupts using the training examples in X. 
%                  [N x M] with N total number of training examples and M
%                  neurons in the output layer
%==========================================================================

% Get total number of examples
N = size(X, 1);
OUT = X;
OUT = [OUT ones(N, 1)]*weights;

end