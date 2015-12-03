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
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
X=[ones(size(X,1), 1) X]; %add ones to X matrix 
hiddenActivation = sigmoid(X*Theta1');
hiddenActivation = [ones(size(hiddenActivation,1),1) hiddenActivation];
htheta = sigmoid(hiddenActivation*Theta2');
for i =1:m
   initialized_y = zeros(num_labels,1);
   initialized_y(y(i)) = 1;
   ksum = 0;
  for k = 1: num_labels
    ksum =ksum + (-initialized_y(k)*log(htheta(i,k)) - (1-initialized_y(k))*log(1-htheta(i,k)));
  end
 J = J+ ksum;


 %J = J+ sum(sum(-initialized_y.*log(htheta(i,:))-(1-initialized_y).*log(1-htheta(i,:)))); % --> this was a trial to get rid of 
 % for loop but not working correctly.

end
J = J/m;
regularization_term  = (sum(sum(Theta1.^2))- sum(sum(Theta1(:,1).^2))) + sum(sum(Theta2.^2))- sum(sum(Theta2(:,1).^2));
regularization_term = (regularization_term*lambda)/(2*m);
J = J + regularization_term;

%======================= End Of Part 1 ===============================

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
activation_1 = zeros(1,size(X,2));
DELTA_1 = 0;
DELTA_2 = 0;
for i = 1:m
 initialized_y = zeros(num_labels,1);
  initialized_y(y(i)) = 1;
  activation_1 =  X(i,:);
  activation_2 = sigmoid(activation_1*Theta1'); 
  activation_2 = [1 activation_2];
  activaton_out = sigmoid(activation_2*Theta2');
  activaton_out = activaton_out';
  delta_3 = activaton_out - initialized_y;
  delta_2 = Theta2' * delta_3 .*  sigmoidGradient([1; Theta1*activation_1']);
  delta_2 = delta_2(2:end);
  Theta1_grad = Theta1_grad + delta_2 * (activation_1);
  Theta2_grad = Theta2_grad + delta_3 *(activation_2);
end
 Theta1_grad = Theta1_grad / m;
 Theta2_grad = Theta2_grad / m;

 %Regularization term added to the given cost functions.
Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + lambda / m * Theta1(:, 2:end);
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + lambda / m * Theta2(:, 2:end);



%======================End Of Part 2 ====================================
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
