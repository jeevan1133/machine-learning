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


                                %      @parameters
  %nn_params --> unrolled vector of Theta so that we could get Theta1 and Theta2
                                %input_layer_size,--> 20 * 20 layer of digits
                                %hidden_layer_size,-->25 hidden units
                                %num_labels, --> 10 labels from 1 to 10
                                %X --> data matrix
                                %y-->output
                                %lambda-->weight regularization parameter
a1 = X ;
a1 = [[ones(size(a1,1),1)],a1] ;
z2=(a1 * Theta1');
a2 = sigmoid(z2);
a2 = [[ones(size(a2,1),1)],a2] ;
z3 = (a2 * Theta2');
hyp = sigmoid(z3) ;

%====This is using for loop. However, we are more interested with vector.
%yVec = 0 ;
%for i=1:m,
%  yVec(i,y(i))=1,
%end
%

y_var = eye(num_labels)(y,:);

f_log=y_var .* log(hyp) ;
s_log = (ones(size(X,1),num_labels) - y_var) .* log(ones(size(X,1),num_labels) - hyp) ;

J = sum(f_log + s_log) ;
J = sum(-J/m) ;

                                %==========Regularization=========
Theta1(:,1) = 0;
Theta2(:,1) = 0 ;
thet_1 =sum(sum(Theta1 .^2));
thet_2 =sum(sum(Theta2 .^2));

reg_term = lambda / (2 * m ) * (thet_1 + thet_2) ;

J  += reg_term ;


%===========Back Propagation Algorithm===========
for t = 1:m,
  d3= hyp - y_var ;
  d_2 = d3 * Theta2(:,2:end);
  d2 = d_2 .* sigmoidGradient(z2) ;

  D1 = d2' * a1 ;
  D2 = d3' * a2 ;

end

Theta1_grad = 1/m * D1 ;
Theta2_grad = 1/m * D2 ;


                %===============Regularized Back Propagation Method=============

Theta1(:,1) = 0 ;
Theta2(:,1) = 0 ;

Theta1_grad = Theta1_grad + lambda / m * Theta1 ;
Theta2_grad = Theta2_grad + lambda /m * Theta2 ;
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
