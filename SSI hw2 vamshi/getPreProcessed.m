function [data_processed, data_mean, data_std] = getPreProcessed( inputdata , type)
% % this function does the preprocessing required for data
% inputs:: --------------------------
%     data       :         data to be preprocessed
%     type       :         preproscessing type
% if:    1- standardized , if: 2 - tranformed   , if:  3 - binarize -------                 
% outputs::  ------------------------
%     data_processed:       gives processed data
%     data_mean   :       gives mean of data for type1
%     data_std    :       gives std of data for type1
% -------------------------------------------


switch type
 % %%   CASE::3 /////////////////////////////////
    case 1 % Here standardising the data
        
        % calculating si - sbar/sigmai
        
        % Here calculating the mean of data
        data_mean = mean(inputdata);
        data_mean_1 = repmat(data_mean,size(inputdata,1),1);
        
        % Here we are calculating standard deviation of data
        data_std = std(inputdata);
        data_std_1 = repmat(data_std,size(inputdata,1),1);
        
        % Subtacting mean and divide by std. dev. to standardise data
        data_processed = (inputdata - data_mean_1)./data_std_1;
  
 %  %% CASE::2 /////////////////////////////////      
    case 2 % Now we have to Transform the features using log(xij + 0.1)
        
        data_processed = log(inputdata + 0.1);
        
 %  %% case:: 3 ///////////////////////////////////////////
    case 3 % to Binarizing the features using I(xij > 0), i.e.
        % make every feature vector a binary vector
        data_processed = inputdata > 0;
end

end

