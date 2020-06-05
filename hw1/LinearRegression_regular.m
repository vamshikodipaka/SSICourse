function [err_rms norm_w]= LinearRegression_regular(X,y,lambda)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function computes the regression parameters with regularisation for M=10 and plots the fitted data
% Use   [error_rms norm_w] = LinearRegress_regular(X,y,lambda) when using to recover errors
% input
%     X - feature data
%     y - output data
%     lambda - regularization_parameter
%
% output
% Example
% LinearRegress_regular(X_train,y_train,0.001);

%% code starts from here

phi = [ones(size(X,1),1)];

% to design matrix
for ii = 1:10
    
    phi = [phi X.^ii];
    
end

temp = pinv((lambda*eye(11)) + (phi'*phi)); % reg parameter included in computation
w = temp * phi' * y; % the regresion parameters computed
y_estimate = w' * phi'; % the output from computed parameters

%% uncomment to check the error and norm for various lambda

err_rms = [];
norm_w = [];

% error estimation 

for jj = linspace(0,lambda,10)
    
    temp = pinv((jj*eye(11)) + (phi'*phi));
    
    w = temp * phi' * y;
    
    y_estimate = w' * phi';
    
    difference = (y_estimate - y').^2;
    difference = sum(difference)/2;
    err_rms = [err_rms sqrt(2*difference/length(X))];
    norm_w = [norm_w norm(w)^2];
    
end

%% to plot the results after parameter computation (comment when using errors)

figure(2)
scatter(X,y)
hold on;
plot(X,y_estimate)
xlabel('input feature');
ylabel('output value');
title('after regularization');