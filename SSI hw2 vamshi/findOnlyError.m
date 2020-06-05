function error_percent = findOnlyError( labels_predicted, labels_true )
% this function is used to compute the error using predicted lables and true labels
% inputs -------------------
%     labels_predicted  :  predicted labels of data
%     labels_truth      :  true labels of data
% outputs --------------------
%     error_percent         :  percentage of error 
% -------------------------------------------------------

% Computing error -------------- 
error = 0;
for no_of_samp = 1:length(labels_predicted)
    if labels_predicted(no_of_samp) ~= labels_true(no_of_samp)
        error = error + 1;
    end
end
error_percent = error/length(labels_predicted)*100;


end

