function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)

% NNCOSTFUNCTION Implements the neural network cost function for a two layer
% neural network which performs classification
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

% Setup some useful variables
m = size(X, 1);


% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



% Part 1: Computing the unregularised cost function

% Add a column of ones to X
X = [ones(m,1) X];


% Compute hidden layer activation units (a2)
z2 = Theta1 * X';
a2 = sigmoid(z2);


% Add a column of ones to a2
t = a2';
t = [ones(m,1) t];


% Compute output layer activation units (a3)
z3 = Theta2 * t';
a3 = sigmoid(z3);


% Recode y from 5000 x 1 to 5000 x 10 matrix
for i = 1:num_labels;
y = [y zeros(m,1)];
endfor

for i = 1:m;
ind = y(i,1);
y(i,ind + 1) = 1;
endfor

y(:,1) = [];


% Compute cost function J
htheta = a3';
K = num_labels;

for i = 1:m;
for j = 1:K;
J = J + (-1 * y(i,j) * log(htheta(i,j)) - ((1 - y(i,j)) * log(1 - htheta(i,j))));
endfor
endfor

J = J/m;



% Compute regularised cost function

% Create non biased Theta1 and Theta2
NBT1 = Theta1;
NBT2 = Theta2;
NBT1(:,1) = [];
NBT2(:,1) = [];

% Compute the regularised term
R1 = (NBT1.^2);
R1 = sum(R1(:));
R2 = (NBT2.^2);
R2 = sum(R2(:));
R  = R1 + R2;
R  = (lambda/(2*m)) * R;
J  = J + R;


% Backpropagation

% X is 5000 x 401
for t = 1:m;
a1 = X(t,:)(:); % 401 x 1
z2 = Theta1 * a1; % Theta1 25 x 401, thus z2 is 25 x 1
a2 = sigmoid(z2);
a2 = [1; a2]; % a2 is now 26 x 1
z3 = Theta2 * a2; % Theta2 is 10x26 so z3 is 10 x 1
a3 = sigmoid(z3);

d3 = a3 - y(t,:)'; % d3 is 10 x 1

d2 = (NBT2' * d3) .* sigmoidGradient(z2); % 25 x 1

Theta1_grad = Theta1_grad + (d2 * a1'); % 25 x 401
Theta2_grad = Theta2_grad + (d3 * a2'); % 10 x 26

endfor

Theta1_grad = Theta1_grad/m;
Theta2_grad = Theta2_grad/m;

% Regularisation after backpropagation

T1 = Theta1(:,1) = []; % 25 x 400
T1 = (lambda/m) * Theta1;
T1 = [zeros(hidden_layer_size,1) T1];
Theta1_grad = Theta1_grad + T1;
                             
T2 = Theta2(:,1) = []; % 10 x 25
T2 = (lambda/m) * Theta2;
T2 = [zeros(num_labels,1) T2];
Theta2_grad = Theta2_grad + T2;
                             


% =============================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];



end
