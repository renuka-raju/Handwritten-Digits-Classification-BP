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
Theta1_grad = zeros(size(Theta1));%25x401
Theta2_grad = zeros(size(Theta2));%10x26
%size(Theta1)
%size(Theta2)
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
X = [ones(m, 1) X];
A2=zeros(size(X,1),size(Theta1,1));
A2=X*Theta1';
% 5000*401 x 401*25 ==>>5000*25
% 5000 examples, each being output to 26 hidden computational unit
A2=sigmoid(A2);
%size(A2)

Output=zeros(size(X,1),size(Theta2,1));
A2 = [ones(m, 1) A2];
Output=A2*Theta2';
% 5000*26 x 26*10-->>5000*10
Output=sigmoid(Output);
%size(Output)

logh_theta1=zeros(m,num_labels);
logh_theta1=log(Output);
y_binary=zeros(m, num_labels);
for i=1:m
y_binary(i,y(i))=1;
end

Aterm=zeros(m);
Aterm=-1*(logh_theta1.*y_binary);
%5000*10 
onevector=ones(m,num_labels);
h_theta2=zeros(m,num_labels);
h_theta2=onevector-Output;
logh_theta2=zeros(m,num_labels);
logh_theta2=log(h_theta2);
%5000*10 
Bterm=zeros(m,num_labels);
Y=onevector-y_binary;
%5000*10 
Bterm=-1*(logh_theta2.*Y);
%5000*10 
total_term=zeros(m);
total_term=sum(Aterm,2)+sum(Bterm,2);%sum along the columns 
%5000*1 
AplusBby_m=sum(total_term)/m;
%1*1
J=AplusBby_m;
% -------------------------------------------------------------
%lambda/2m sum(square of theta j=1:n)
%overfitting term
theta1_square=zeros(size(Theta1),1);
theta1_square=power(Theta1,2);
sumtheta1_square=sum(sum(theta1_square,2)-sum(theta1_square(1),2));
theta2_square=zeros(size(Theta2),1);
theta2_square=power(Theta2,2);
sumtheta2_square=sum(sum(theta2_square,2)-sum(theta2_square(1),2));
Cterm=(lambda/(2*m))*(sumtheta1_square+sumtheta2_square);
%Cterm - regularization term
J=AplusBby_m+ Cterm;

% =========================================================================

% Unroll gradients 
del_output=Output-y_binary;
%5000 samples x 10 output units for each sample
%A2 already is the sigmod function of (X*Theta1')
A2forgrad=X*Theta1';
A2forgrad = [ones(m, 1) A2forgrad];
for p=1:m
del_hidden=(Theta2'*del_output(p,:)').*sigmoidGradient(A2forgrad(p,:)');%(26*10x10*1).26*1--26*1
del_hidden=del_hidden(2:end);%25*1
%del_hidden
%size(del_hidden)
Theta2_grad=Theta2_grad+(del_output(p,:)'*A2(p,:));%10*1 x 1*26--10x26
%size(Theta2_grad)
%size(Theta1_grad)
%size(X(p,:))
Theta1_grad=Theta1_grad+(del_hidden*X(p,:));%25*1 x 1*401--25*401
end
%Theta1_grad=(1/m)*Theta1_grad+(lambda/m)*(Theta1);%25*401 /// 5x4
%Theta2_grad=(1/m)*Theta2_grad+(lambda/m)*(Theta2);%10x26  /// 3x6
Theta2_grad(:,1)=(1/m)*Theta2_grad(:,1);
Theta1_grad(:,1)=(1/m)*Theta1_grad(:,1);
%adding regularization term for 5x3 (ist column bias term)
Theta1_grad(:,2:end)=((1/m)*Theta1_grad(:,2:end))+((lambda/m)*(Theta1(:,2:end)));
%adding regularization term for 3x5 (ist column bias term)
Theta2_grad(:,2:end)=((1/m)*Theta2_grad(:,2:end))+((lambda/m)*(Theta2(:,2:end)));
grad = [Theta1_grad(:) ; Theta2_grad(:)];%(38x1 from checkNNGradients theta1 5x4 theta2 3x6)
%size(grad)
end
