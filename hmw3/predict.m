function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);
X=[ones(size(X,1), 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

hidden = X*Theta1'; % to calculate the weighted som of given features 

hidden = sigmoid(hidden); % to calculate the activation of each hidden unit

hidden = [ones(size(hidden,1),1) hidden];

htheta = hidden * Theta2'; % to calculate the weighted sum of hidden units 

htheta = sigmoid (htheta); % to calculate the each units activation in terms of given weighted sums.

% Please notice that hteta is a 5000x10 matrix that contains 10 possibilities for each example and since we have
% 5000 example every row contains different values of example. 

[temp, p] = max(htheta, [], 2);

 






% =========================================================================


end
