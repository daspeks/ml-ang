function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, num_features, lambda)

% COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%


% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%

% num_movies        5
% size(X_grad)      5 x 3
% num_users         4
% size(Theta_grad)  4 x 3

T = ((Theta * X' - Y').^2)';
T = R .* T;
J = (1/2) * sum(T(:));


for i = 1:num_movies
    % Find users that rated movie i - returns a row vector or a single element
    idx = find(R(i, :) == 1);

    % Finds Theta parameters that correspond to users idx - returns [idx x 3] matrix
    Theta_temp = Theta(idx, :);

    % Finds Y parameters that correspond to users idx - returns [1 x idx] matrix
    Y_temp = Y(i, idx);

    % Find X_grad
    X_grad(i, :) = ((X(i, :) * Theta_temp' - Y_temp) * Theta_temp) + lambda * (X(i,:));
end


for j = 1:num_users

    % Find movies rated by user 1
    idx = find(R(:, j) == 1);

    Theta_temp = Theta(j, :);     % 1 x 3
    X_temp     = X(idx, :);       % 5 x 3
    Y_temp     = Y(idx, j);       % 5 x 1

    Theta_grad(j, :) =  ((X_temp * Theta_temp' - Y_temp)' * X_temp) + lambda * Theta(j, :);
end


                         
% Regularized cost function
J = J + ((lambda/2) * (sum(Theta(:).^2) + sum(X(:).^2)));

% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
