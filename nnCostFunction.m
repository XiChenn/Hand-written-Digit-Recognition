function [J grad] = nnCostFunction(nn_params, ...
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
%   lambda is not used - we're keeping it here so the two cost functions can be
%   sent to the same gradient checking code later 
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
%y2 = zeros(m, num_labels); %5000x10
%for i = 1:m
%    y2(i, y(i)) = 1;
%end
J = sum(sum((log(h') .* y + log(1 - h') .* (1 - y)) / -m));
%for i = 1:num_labels
%    J += (log(h(:, i)') * y(i, :)' + log(1 - h(:, i)') * (1 - y(i, :)')) / -m;
%end

%---------------------------------------------------
% Back Propagation algorithm - non-regularized
%---------------------------------------------------
delta3 = h' - y; % 10x5000
delta2 = Theta2' * delta3 .* (A2' .* (1 - A2')); % 26x5000
Theta1_grad = delta2(2:end, :) * X ./m;	%Theta1_grad: 25*401; Remove first row in delta2
Theta2_grad = delta3 * A2 ./m;			%Theta2_grad: 10*26
% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];



end
