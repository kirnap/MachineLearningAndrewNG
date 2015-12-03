%Test File 

hidden = X*Theta1'; % to calculate the weighted som of given features 

hidden = sigmoid(hidden); % to calculate the activation of each hidden unit

hidden = [ones(size(hidden,1),1) hidden];

htheta = hidden * Theta2'; % to calculate the weighted sum of hidden units 

htheta = sigmoid (htheta);