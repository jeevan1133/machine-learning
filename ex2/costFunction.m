function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
%
% Note: grad should have the same dimensions as theta
%
%===================Cost Function==================%
%
%    The formula for J is 1/m(-y' * log(sigmoid(z)) - ((1-y)'* log(1-h)))
z = X * theta ;
h = sigmoid(z);
log_first_value = (-y' * log(h));
log_second_value = ((1-y)' * log(1-h));
log_value = log_first_value - log_second_value ;
J = log_value / m ;

                %==========================Gradient Descent=====================
              % Gradient Descent vectorized form is (1/m * X' * (g(X*theta) -y))
% Note here that there is no alpha present. We are repeating this step without alpha

grad = (1/m * X' * (h - y)) ;


% =============================================================

end
