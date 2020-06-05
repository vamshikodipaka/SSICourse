% this is the main function of the homework from which we can access every
% subpart. There are two functions 
% LinearRegress.m (without regularization)
% LinearRegress_regular.m (with regularization)
% The code is commented for each subparts, to access them pls uncomment the
% required parts.

clc; close all; clear all;

%% to load train data

filename = './hw1training.txt'; 
Data1 = load(filename);            % 10X2 matrix 
X_training = Data1(:,1);
y_training = Data1(:,2);
M =10; 

%% to load test data

filename = './hw1test.txt';
Data2 = load(filename);             % 100x2 matrix
X_testing = Data2(:,1);
y_testing = Data2(:,2); 



%% for a fixed M = 10 to compute linear regression and plot the results

LinearRegression(X_training,y_training,M);

% uncomment for training and test error for different values of M without regularization

errrmstrain = [];
errrmstest = [];

for i=1:M
    
errrmstrain = [errrmstrain  LinearRegression(X_training,y_training,i)];
errrmstest = [errrmstest  LinearRegression(X_testing,y_testing,i)];

end

% uncomment for training and test error for different values of lambda with regularization

lambda = 0.001; % the functions will linearly space lambda from 0 to specified value
[errrmstrain norm_w_train] = LinearRegression_regular(X_training,y_training,lambda);
[errrmstest norm_w_test] = LinearRegression_regular(X_testing,y_testing,lambda);


%% fixed reg parameter = 0.001 and results of it
% 
lambda = 0.001;
LinearRegression_regular(X_training,y_training,lambda);

% uncomment to plot training errors
figure(3)
%x_ax = 1:M; % for without regularization
x_ax = linspace(0,lambda,10); % for regularization  1X10 matrix
plot(x_ax,errrmstrain,'r-',x_ax,errrmstest,'g-');
axis([0,lambda,0,1])
xlabel('lambda');
ylabel('RMS Error');
legend('train data','test data');

% uncomment to plot test errors
figure(4)
x_ax = linspace(0,lambda,10); % for regularization
plot(x_ax,norm_w_train,'r-',x_ax,norm_w_test,'g-');
% axis([0,lambda,-20,2e13])
xlabel('lambda');
ylabel('norm');
legend('train data','test data');