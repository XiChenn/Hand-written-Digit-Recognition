function [J grad] = nnRCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));


         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

m = size(X, 1);
% Add bias to X
X = [ones(m, 1), X]; %5000x401
A2 = sigmoid(X * Theta1'); %5000x25
A2 = [ones(size(A2, 1), 1), A2]; %5000x26
h = sigmoid(A2 * Theta2'); %5000x10

% Turn y to a 10x5000 matrix
y = eye(num_labels)(:, y); % 10x5000

J = sum(sum((log(h') .* y + log(1 - h') .* (1 - y)) / -m));

sum1 = sum(sum(Theta1(: , 2:end) .^2)); %not caculate the first column of Theta1
sum2 = sum(sum(Theta2(: , 2:end) .^2));

J = J + (lambda / (2 * m)) * (sum1 + sum2);	%with regulation; 

%---------------------------------------------------
% Back Propagation algorithm - regularized
%---------------------------------------------------
delta3 = h' - y; % 10x5000
delta2 = Theta2' * delta3 .* (A2' .* (1 - A2')); % 26x5000
Theta1_grad = delta2(2:end, :) * X ./m;	%Theta1_grad: 25*401; Remove first row in delta2
Theta2_grad = delta3 * A2 ./m;			%Theta2_grad: 10*26
Theta1_grad += (lambda / m) * [zeros(size(Theta1, 1), 1), Theta1(:, 2:end)];	%set the first column of Theta1 to zeros.
Theta2_grad += (lambda / m) * [zeros(size(Theta2, 1), 1), Theta2(:, 2:end)];

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
