function [J, grad] = linearRegCostFunction(X, y, theta, lambda)

% LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear
% regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% Compute cost function J
ht = X * theta;
t = ht - y;
J = sum(t.^2) / (2 * m);
u = theta.^2;
u(1) = 0;
J = J + ((lambda/(2*m))*sum(u(:)));


% Compute gradient descent
grad = (1/m) * (X' * t);
u = (lambda/m) * theta;
u(1,1) = 0;
grad = grad + u;

                
% =========================================================================

grad = grad(:);

end
