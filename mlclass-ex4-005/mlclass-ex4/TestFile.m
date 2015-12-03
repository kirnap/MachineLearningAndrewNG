%Test File
%This file is written to predict the values from trained weights
%Result matrix htheta contains 5000x10 each has 10 predictions for 10 different output units and there are
%5000 different training examples

%X=[ones(size(X,1), 1) X];

%hidden = X*Theta1'; % to calculate the weighted som of given features 

%hidden = sigmoid(hidden); % to calculate the activation of each hidden unit

%hidden = [ones(size(hidden,1),1) hidden];

%htheta = hidden * Theta2'; % to calculate the weighted sum of hidden units 

%htheta = sigmoid (htheta);
X=[ones(size(X,1), 1) X];

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
  delta_2 = Theta2' * delta_3 .* [1; sigmoidGradient(Theta1*activation_1')];
  delta_2 = delta_2(2:end);
  DELTA_1 = DELTA_1 + delta_2 * (activation_1);
  DELTA_2 = DELTA_2 + delta_3 *(activation_2);
end
accumulated_Gradiet_1 = (1/m)* DELTA_1;
accumulated_Gradiet_2 = (1/m)* DELTA_2;