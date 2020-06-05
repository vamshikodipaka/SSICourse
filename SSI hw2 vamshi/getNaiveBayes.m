function test_lab_naive = getNaiveBayes( data_training, data_train_label, data_testing )
% this function is used to get Naive Bayes classification when given a testing data
% ------------------ NAIVES BAYES IMPLEMENTATION ---------------------
% input: --------------
%       data_training     - training data
%       data_train_label - training data labels
%       data_testing      - testing data to classify
%       
% output: ----------------------------------  
%       label   - label of test data after classification
% ------------------------------------------------------

%%  WE CAN ALSO USE INBUIT FUNCTIONS OF MATLAB TO FIND TRAIN AND TEST ERRORS OF NAIVE BAYES
% 
% Naive Bayes Classifier for training data
% NBModel = fitcnb(norm_train_features, ytrain);
% Pred_Class_NB = resubPredict(NBModel);
% C_train_NB = confusionmat(ytrain,Pred_Class_NB);
% correct_classified_NB = C_train_NB(1,1) + C_train_NB(2,2);
% train_spam_accuracy_NB = correct_classified_NB*100 / 3065;
% %----------------------------------------
% % Using Naive Bayes model to fit the test data
% Pred_Class_test_NB = predict(NBModel, norm_test_features);
% C_test_NB = confusionmat(ytest,Pred_Class_test_NB);
% correct_classified_test_NB = C_test_NB(1,1) + C_test_NB(2,2);
% test_spam_accuracy_NB = correct_classified_test_NB*100 / 1536;%%

%%
% //////Finding the probabilities of two classes Spam and Notspam //////
prob_spam = nnz(data_train_label)/numel(data_train_label);
prob_not_spam = 1 - prob_spam;

% /////// Finding mean and standard deviation of training fetures for class Spam and notSpam//////
spam_indices = find(data_train_label);  % find spam indices
mean_spam = mean(data_training(spam_indices,:)); % do mean and std dev for spam data
std_spam = std(data_training(spam_indices,:));

not_spam_indices  =find(~data_train_label); % find notspam indices
mean_not_spam = mean(data_training(not_spam_indices,:)); % do mean and std dev for spam data
std_not_spam = std(data_training(not_spam_indices,:));

%%
% finding the probabilities of each feature for class 1 Spam of test data -----
prob_feature_spam_final = [];
for ndata = 1 : size(data_testing,1)
    prob_feature_spam = getProbabilityFeature( data_testing(ndata, :), mean_spam, std_spam );
    prob_feature_spam_final = [prob_feature_spam_final ; prod(prob_feature_spam) * prob_spam];
end

% finding the probabilities of each feature for class 2 NonSpam of test data ------
prob_feature_nonspam_final = [];
for ndata = 1 : size(data_testing,1)
    prob_feature_not_spam = getProbabilityFeature( data_testing(ndata,:), mean_not_spam, std_not_spam );
    prob_feature_nonspam_final = [prob_feature_nonspam_final ; prod(prob_feature_not_spam) * prob_not_spam];
end
%%
% check for higher probabilities to assign it to a class
test_lab_naive =  prob_feature_spam_final > prob_feature_nonspam_final;

end


