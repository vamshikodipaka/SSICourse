function error_rms = LinearRegression(X,y,M)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function computes the regression parameters and plots the fitted data
% Use  error_rms = LinearRegress(X,y,M) when using to recover errors
% input:
%     X - feature data
%     y - output data
%     M - polynomial order to fit
%
% output:
%     it displays the figure plot
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Example
% LinearRegress(X_training,y_training,10);

%% code starts from here

phi = [ones(size(X,1),1)];

% to form design matrix
for i = 1:M
    
    phi = [phi X.^i];
    
end

temp = pinv(phi'*phi);
w = temp * phi' * y; % the regresion parameters computed
y_estimate = w' * phi'; % the output from computed parameters

%% compute the RMS error
difference = (y_estimate - y').^2;
difference = sum(difference)/2;
error_rms = sqrt(2*difference/length(X));

%% results after parameter computation
figure(1)
scatter(X,y)
hold on;
plot(X,y_estimate)
xlabel('input feature');
ylabel('output value');
title('before regularization');