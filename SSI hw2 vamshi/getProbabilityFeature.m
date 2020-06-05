function probability_feature = getProbabilityFeature( feature, mean1, std1 )
 %%=====================================================
% this function outputs probability of a feature of a given class
% input============
%     feature::    features to find probability
%     mean1   ::            mean of train data
%     std1    ::            std of train data
%                         
% output
%     prob_feature   ::        Probability of feature 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

probability_feature = exp((-(feature - mean1).^2)./(2.*std1.^2))./(sqrt(2 * pi) .* std1);
end

