%% Adapted from Machine Learning Online Class - Exercise 4 Neural Network Learning




%% Initialization
clear ; close all; clc

%% Setup the parameters you will use for this exercise
input_layer_size  = 400;  % 20x20 Input Images of Digits
hidden_layer_size = 25;   % 25 hidden units
num_labels = 10;          % 10 labels, from 1 to 10   
                          % (note that we have mapped "0" to label 10)



% Load Training Data
fprintf('Loading Data ...\n')

load('mnistData.mat');
m = size(X, 1);


% Load the weights into variables Theta1 and Theta2
load('nnWeights.mat');

% Unroll parameters 
nn_params = [Theta1(:) ; Theta2(:)];

%% Implement the feed-forward part that returns the cost.  You will have to do 
%  an unregularized and regularized cost. 
%
fprintf('\nFeedforward: non-regularized cost\n')


J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, ...
                   num_labels, X, y);

fprintf(['Cost at parameters (loaded from nnWeights): %f '...
         '\n(this value should be about 0.287629)\n'], J);

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% Implement the regularized cost
%

fprintf('\nRegularized cost function\n')

% Weight regularization parameter (we set this to 1 here).
lambda = 1;

J = nnRCostFunction(nn_params, input_layer_size, hidden_layer_size, ...
                   num_labels, X, y, lambda);

fprintf(['Cost at parameters (loaded from nn weights): %f '...
         '\n(this value should be about 0.383770)\n'], J);

fprintf('Program paused. Press enter to continue.\n');
pause;


%  Randomly initialize the NN parameter weights.

fprintf('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];


%% Backpropagation
%  Implement the NN gradients (the partial derivatives of the parameters).
%  You can check these by also calculating the gradients analytically and comparing
%  the results. First do this for the non-regularized version
%  NOTE: before make sure you've looked at the HW instructions telling you to
%  code up the analytical gradient method!
%
fprintf('\nChecking Gradients... \n');

%  Check gradients by running checkNNGradients
checkNNGradients;

fprintf('\nProgram paused. Press enter to continue.\n');
pause;


%  Now compute the regularized version of the gradients
%

fprintf('\nChecking Gradients (w/ Regularization) ... \n')

%  Check gradients by running checkNNGradientsR
lambda = 3;
checkNNGradientsR(lambda);

% Also output the costFunction debugging values
debug_J  = nnRCostFunction(nn_params, input_layer_size, ...
                          hidden_layer_size, num_labels, X, y, lambda);

fprintf(['\n\nCost at (fixed) debugging parameters: %f ' ...
         '\n(this value should be about 0.576051)\n\n'], debug_J);

fprintf('Program paused. Press enter to continue.\n');
pause;


%% use fmincg to train the NN
%
fprintf('\nTraining Neural Network... \n')

%  After you have completed the assignment, change the MaxIter to a larger
%  value to see how more training helps.
options = optimset('MaxIter', 50);

%  You should also try different values of lambda
lambda = 1;

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnRCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

fprintf('Program paused. Press enter to continue.\n');
pause;



%% Implement your predict function to check the accuracy of your model.

pred = predict(Theta1, Theta2, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);


