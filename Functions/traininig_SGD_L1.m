function [weights] = traininig_SGD_L1(X, Y, weights,numIter,RegularizationRate,LearningRate)
% traininig of the neural network with SGD-L1
%==========================================================================
%Input: X--> training matrix [N x M*IP] with N number of observation and
%            M*P total number or regressors (time lagged variables)
%
%       Y--> training outputs [N x M] with N observations and M number of
%            neurons in the ouput layer
%
%      numIter - Number of iterations Stochastic Gradient Descent
%                should take while training
%
%Outputs: weigths--> matrix of weights of the network corresponding to the
%                    parameters of a VAR model
%
%==========================================================================
% Get batch size
% Remove decimal places in case of improper input
B = floor(size(X,1)); % number of observations for the full batch size

% First cell array is for the initial
xNeuron = cell(1, 2);
% For L1 regularization
% This represents the total L1 penalty that each
% weight could have received up to current point
uk = 0;
% Total penalty for each weight that was received up to
% current point
qk = zeros(size(weights));
xNeuron{1} = [X ones(B, 1)];
% For each iteration...
for ii = 1 : numIter
    
    %Perform forward propagation
    HH=xNeuron{1,1};
    sNeuron=HH*weights;
    % Compute outputs of this layer
    xNeuron{1,2} = sNeuron;
    %%% Perform backpropagation
    % Compute sensitivities for output layer
    delta = (xNeuron{end} - Y) .* ones(size(sNeuron));
    
    %%% Compute weight updates
    alpha = LearningRate;
    lambda = RegularizationRate;
    
    % Obtain the outputs and sensitivities for each
    % affected layer
    XX = xNeuron{1,1};
    D = delta;
    
    % Calculate batch weight update
    C=XX.'*D;
    weight_update=(1/B)*C;
    % Compute the final update
    weights = weights - alpha * weight_update;
    % Apply L1 regularization if required
    
    % Step #1 - Accumulate total L1 penalty that each
    % weight could have received up to this point
    uk = uk + (alpha * lambda / B);
    
    % Step #2
    % Using the updated weights, now apply the penalties
    % 2a - Save previous weights and penalties
    % Make sure to remove bias terms
    z = weights(1 : end - 1,:);
    q = qk(1 : end - 1,:);
    % 2b - Using the previous weights, find the weights
    % that are positive and negative
    w = z;
    indwp = w > 0;
    indwn = w < 0;
    
    % 2c - Perform the update on each condition
    % individually
    w(indwp) = max(0, w(indwp) - (uk + q(indwp)));
    w(indwn) = min(0, w(indwn) + (uk - q(indwn)));
    
    % 2d - Update the actual penalties
    qk(1:end-1,:) = q + (w - z);
    
    % Don't forget to update the actual weights!
    weights(1 : end - 1, :) = w;
    
    
end
end

