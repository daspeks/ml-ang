function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


n = length(theta); % number of parameters

% Compute hypothesis model -- same dimension as X
h_theta = sigmoid(theta' * X');

% Compute cost summation term
term = -y .* log(h_theta') - (1 - y) .* log (1 - h_theta');

% Compute cost
J = sum(term)/m;

% Add regularised term to cost
J = J + ((lambda/(2*m)) * sum(theta(2:n).^2));

% Compute gradient summation term
grad = (h_theta' - y)' * X;
grad = grad / m;

%
temp = lambda/m;
for i = 2:n;
        grad(i) = grad(i) + (temp * theta(i));
endfor


% =============================================================

end
