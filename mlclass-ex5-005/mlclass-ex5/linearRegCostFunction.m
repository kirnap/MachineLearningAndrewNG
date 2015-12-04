function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
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

h_theta = X*theta;
J= (sum((h_theta - y).^2))/(2*m) + (sum(theta(2:end).^2))*(lambda/(2*m));



grad = 1 / m * X' * (h_theta - y); %This part is thougth during the third exercise
grad(2:end) = grad(2:end)+(lambda/m)*theta(2:end);

%================================problematic part ==========================================
% In this part of the code I tried to implement grad function from the mathematical formula of it but I encountered with A(I) = X
%must have the same lengths and that is why I ignored this part and tried the one in above and it works correctly.
%This portion of the code is bugy for the future use that is why I've written the previous part to recover bugs.
% grad(1) = sum((h_theta-y).*X(:,1))/m;
% grad(2:end) = (sum((h_theta-y).*X(:,2:end))/m)+(lambda/m)*theta(2:end);


% =========================================================================

grad = grad(:);

end
