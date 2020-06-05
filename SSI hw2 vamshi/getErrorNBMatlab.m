function getErrorNBMatlab()
% \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
% function gets training and test errors using Naive Bayes classifier of MATLAB
% 
% datais read from the current working directory
% ///////////////////////////////////////////////////////

clc; clear all; close all;

% to load training data
data_training = load('./spamTrain.txt');
data_training_lab = load('./spamTrainLabels.txt');

% to load testing data
data_testing = load('./spamTest.txt');
data_testing_lab = load('./spamTestLabels.txt');

%% 1.3.a Standardizing the columns so they all have mean 0 and unit variance
[data_stand, data_mean_train, data_std_train] = getPreProcessed( data_training , 1); % type 1 for standardise

% preprocessing testing data using TESTING data mean and standard variation
data_testing_stand = getPreProcessed( data_testing , 1); % preprocessing the test data 


%% 1.3.b Transform the features using log(xij + 0.1)
data_trans = getPreProcessed( data_training , 2); % type 2 for transforming features
data_testing_trans = getPreProcessed( data_testing , 2); % to preprocessing the test data

%% 1.3.c Binarize the features using I(xij > 0), i.e.
% make every feature vector a binary vector

data_bin = getPreProcessed( data_training , 3); %type 3 for binarize
data_test_bin = getPreProcessed( data_testing , 3); % to preproces the test data


%% gettting the probabilities of two classes Spam and Notspam
probab_spam = nnz(data_training_lab)/numel(data_training_lab);
probab_notspam = 1 - probab_spam;

%% Using standardise data

% Building classifier
prior = [probab_spam probab_notspam];
classNames = [1 0];
Model_stand = fitcnb(data_stand, data_training_lab, 'DistributionNames', 'normal', 'ClassNames' , classNames, 'Prior' , prior);
%-------------
%predicting test class
test_lab_NBMatlab = predict(Model_stand, data_testing_stand);

% get error----------
%to compute error of Test data
err_stand_naive = findOnlyError( test_lab_NBMatlab, data_testing_lab );
disp('==========================================================');
disp('Naive Bayes Classifier(Matlab) Results');
disp(['Error in testing set using standardised data is ',num2str(err_stand_naive)]);
disp('=======================================================');
%% Using Transformed data

% Building classifier
prior = [probab_spam probab_notspam];
classNames = [1 0];
Model_trans = fitcnb(data_trans, data_training_lab, 'DistributionNames', 'normal', 'ClassNames' , classNames, 'Prior' , prior);

%predicting test class ----------
test_lab_NBMatlab = predict(Model_trans, data_testing_trans);

% getting error to compute error of Test data ---------------
error_trans_naive = findOnlyError( test_lab_NBMatlab, data_testing_lab );
disp('===================================');
disp('Naive Bayes Classifier(Matlab) Result');
disp(['Error in test set using Transformed data is ',num2str(error_trans_naive)]);
disp('====================================');

%% Using Binarized data :: Temporarily the below function is not working

% % Build classifier
% prior = [prob_spam prob_notspam];
% classNames = [1 0];
% Model_bin = fitcnb(double(data_bin), data_train_lab, 'ClassNames' , classNames, 'Prior' , prior);
% 
% %predict test class
% test_lab_NBMatlab = predict(Model_bin, data_test_bin);
% 
% % get error
% %to compute error of Test data
% error_bin_naive = getOnlyError( test_lab_NBMatlab, data_test_lab );
% disp('*******************************************************************');
% disp('Naive Bayes Classifier(Matlab) Result');
% disp(['Error in test set using Binarized data is ',num2str(error_bin_naive)]);
% disp('*******************************************************************');
