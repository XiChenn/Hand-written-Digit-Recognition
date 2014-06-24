function numgrad = computeNumericalGradient(J, theta)
% Computer a numerical estimate of the gradient of function J around theta. 
% 
% theta is a single vector - the NN parameters unrolled. 
% J is your cost function - calling it from here will call it with the
% parameters set up in the calling program plus your theta (the nnParams).
%
% Use e=1e-4 for epsilon 
%
% This function takes in the vector theta unrolled and will output
% a vector of the same size giving the numeric gradient approximations.
% In this case, 
% gradApprox = (J(theta + epsilonV) – J(theta – epsilonV)) /(2*epsilon)
% 
% theta is theta, J is J. epsilonV is a vector where, for each element of theta,
% say index p (see for loop below) the vector is zero except for having 
% epsilon in position p. 
              
% initialize numgrad to zeros
numgrad = zeros(size(theta));
% initialize your EPSILON vector to zeros
epsilonV = zeros(size(theta));

e = 1e-4;

% For each element in your numergrad, calculate its approximation 
for p = 1:numel(theta)

    epsilonV(p) = e;
    loss1 = J(theta - epsilonV);
    loss2 = J(theta + epsilonV);

    numgrad(p) = (loss2 - loss1) / (2*e);
    epsilonV(p) = 0;
end

end
