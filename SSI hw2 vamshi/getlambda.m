function  lambda  = getlambda( data_training, data_training_lab )
% %this function gets lambda from cross validation
% cross validating  data division of training dataset
% Usually 80% for training data and 20% for validation data
% input ---------------------------
%     data_training          training data
%     data_training_lab      training data labels  
% output --------------------
%     lambda              optimal lambda
% --------------------------------------------------

num_training = ceil(0.8 * size(data_training,1)); % selecting 80% of data
data_train_train = data_training(1:num_training,:); % first 80% as training dataset's data
data_train_valid = data_training(num_training+1:end,:); % last 20% as validate data set's data
data_train_lab_train = data_training_lab(1:num_training,:); % first 80% as training data labels
data_train_lab_valid = data_training_lab(num_training+1:end,:);% last 20% as validate data labels

error_percent = zeros(1,11); % to allocate error values
error_index = 1; % index used to store error of validation
% lambda_init= linspace(0,0.05,11); % to parametr sweep lambda
% lambda_init = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 50];
lambda_init = linspace(0,0.1,11);

% to parametr sweep lambda to check for minimum validation error
for lambda = lambda_init
    
    disp(['getting error for lambda = ',num2str(lambda)])
    
    % getting regression parameters using optimisation
    regs_para = getRegPara( data_train_train, data_train_lab_train, lambda );
    
    % predicting the output of validation data using estimated parameters above
    [y_predi] = predict(data_train_valid, regs_para);
    
    y_predi = y_predi > 0.5; % to check and assign a class for data
    
    
    % predicting the training set using predict function
    [training_pred] = predict(data_train_train, regs_para);
    
    training_pred = training_pred > 0.5;  % to check and assign a class for data
    
    % get train and valid errors
    training_error(error_index) = getOnlyError(training_pred, data_train_lab_train);
    
    validing_error(error_index) = getOnlyError(y_predi, data_train_lab_valid);
    

    error_index = error_index + 1;
end
% checking for minimum error and corresponding lambda wil be output
training_error
validing_error
[~,index] = min(validing_error)
lambda = lambda_init(index);
figure
plot(lambda_init,training_error,'r',lambda_init, validing_error,'b');
xlabel('lambda');
ylabel('error (in %)');
title('training/validation error vs lambda');
legend('train error','validation error');
end
