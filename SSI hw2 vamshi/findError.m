function error_percent = findError( data, labels, regs_param)
% % this function is used to get error when data and labels are given
% inputs: -----------
%     data                data with feaures
%     labels              labels of data
%     regs_para           regression parameters               
% outputs: ----------- 
%     error_per    : we get percentage of error 
% --------------------------

% to predict the output of validation data using estimated parameters
[y_predict] = predict(data, regs_param);

y_predict = y_predict > 0.5; % to check and assign a class for data

% Computing validation error for the data
error = 0;
for nsamp = 1:length(labels)
    if labels(nsamp) ~= y_predict(nsamp)
        error = error + 1;
    end
end
error_percent = error/length(labels)*100;
end

