function p = predict(Theta1, Theta2, X)
%Given Theta1 and Theta2, predict the results for matrix X

% Initialize p to contain the predictions 
p = zeros(size(X, 1), 1); %5000x1
% Add your column of ones to X!
X = [ones(size(X, 1), 1), X];
% Use feed-forward propagation to predict your output. 
% Don't forget the bias units!
A2 = sigmoid(X * Theta1'); %5000x25
A2 = [ones(size(A2, 1), 1), A2]; % Add bias units: 5000x26
A3 = sigmoid(A2 * Theta2'); %5000x10
[c, p] = max(A3, [], 2);
% =========================================================================


end
