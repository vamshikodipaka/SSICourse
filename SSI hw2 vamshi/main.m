%% MAIN.M:;: RUN IT INORDER TO GET THE OUTPUTS FOR THE SPAM CLASSIFIER 
% Logistic Regression and Linear Discriminant Analysis
% Uncomment when annd where-ever necessary------------
%% =================== Program starts here ===================

clc; 
clear all; 
close all;

% Firstly, load training data
data_training = load('./spamTrain.txt');
data_training_labels = load('./spamTrainLabels.txt');

% Now, load test data
data_testing = load('./spamTest.txt');
data_testing_labels = load('./spamTestLabels.txt');

%% 1.1 To find Max and mean of the average length of uninterrupted sequences 
% of capital letters in the training set

[ data_max, data_mean ] = getMaxMeanLength( data_training, 55 );

disp('-------------%%%%%%%%%%%%%%%%%%%%%%%%%----------');
disp(['Maximum of the average length of uninterrupted sequences ' ...
    'of capital letters in the training set is ',num2str(data_max)]);
disp(['Mean of the average length of uninterrupted sequences ' ...
    'of capital letters in the training set is ',num2str(data_mean)]);
disp('------------%%%%%%%%%%%%%%%%%%%%%%%%%%%----------');

%% 1.2 To find max and mean of the lengths of the longest uninterrupted sequences
% of capital letters in the training set

[ data_max, data_mean ] = getMaxMeanLength( data_training, 56 );

disp('--------------%%%%%%%%%%%%%%%%%%%%%%-----------');
disp(['Maximum of the lengths of the longest uninterrupted sequences '...
    'of capital letters in the training set is ',num2str(data_max)]);
disp(['Mean of the lengths of the longest uninterrupted sequences ' ...
    'of capital letters in the training set is ',num2str(data_mean)]);
disp('----------------%%%%%%%%%%%%%%%%%%%%%%%%%%%%--------------');

%% 1.3.a Task to Standardize the columns so they all have mean 0 and unit variance
[data_stand_1, data_mean_training, data_std_training] = getPreProcessed( [data_training;data_testing] , 1);
% Here we are standardizing for type 1 

% to preprocess test data using TEST data mean and standard deviation
% data_test_stand = getPreProcessed( data_test , 1); % to preprocess the test data 
data_train_standize = data_stand_1(1:size(data_training,1),:);
data_test_standize = data_stand_1(size(data_training,1)+1:end, :);

%% 1.3.b Task to Transform the features using log(xij + 0.1)
data_train_transf = getPreProcessed( data_training , 2); % type 2 for transform features
data_test_transf = getPreProcessed( data_testing , 2); % to preproces the test data

%% 1.3.c Task to Binarize the features using I(xij > 0), i.e.
% make every feature vector a binary vector

data_binary = getPreProcessed( data_training , 3); %type 3 for binarize
data_test_binary = getPreProcessed( data_testing , 3); % to preproces the test data



%% %%%%%%//////////// NOW WE DO LOGISTIC REG FROM HERE ///////%%%%%%%%%%%%%

%% I. -------  logistic regression with standardised data ------------

data_stand_logis = [ones(size(data_train_standize,1),1) data_train_standize];% to add ones to first column of data

% % Adding row of ones -------------------------
% data_test_standize = [ones(size(data_test_standize,1),1) data_test_standize];
% 
% /////////////////////////////////////////-------------
% % Now, we do preprocessing of TESTING data using TRAINING data mean and standard deviation
% /////////////////////////////////----------------
% data_mean_training = repmat(data_mean_training, size(data_testing,1),1);
% data_std_training = repmat(data_std_training,size(data_testing,1),1);
% data_test_stand_logis = (data_testing - data_mean_training)./data_std_training;

data_test_stand_logis = [ones(size(data_test_standize,1),1) data_test_standize];


%% Choosing lambda basing on minimum cross validation errors -----------------
% lambda  = getlambda( data_stand_logis, data_training_labels )
lambda = 0.01; % i got lambda = 0.01 after minimizing the validation error

%% to get training set error ----------------
regs_param = findRegParam( data_stand_logis, data_training_labels, lambda );
error_train_stand = findError( data_stand_logis, data_training_labels, regs_param);
disp('//////////////////////////////////////////////////////////////');
disp(['./I. Error in training set using standardised data is ',num2str(error_train_stand)]);

% to get test set error -------------------------
error_test_stand =  findError( data_test_stand_logis, data_testing_labels, regs_param);
disp(['./I. Error in test set is ',num2str(error_test_stand)]);
disp('/////////////////////////////////////////////////////////////');


%% II. ---- Logistic regression with Transformed feature data -------

% to add ones to first column of data--------------------
data_trans_logis = [ones(size(data_train_transf,1),1) data_train_transf];

% add rows on ones----------------------
data_test_trans_logis = [ones(size(data_test_transf,1),1) data_test_transf];

% to get lambda basing on minimum validation error-------------
%lambda  = getlambda( data_trans_logis, data_training_labels )
lambda = 0.005; % i got lambda = 0.005 after minimizing the validation error-----

%to get training set error----------------
regs_param = findRegParam( data_trans_logis, data_training_labels, lambda );
error_train_trans = findError( data_trans_logis, data_training_labels, regs_param);
disp('///////////////////////////////////////////////////////////');
disp(['II. Error in training set using transformed data is ',num2str(error_train_trans)]);

% to get test set error------------------------
error_test_trans =  findError( data_test_trans_logis, data_testing_labels, regs_param);
disp(['II. Error in testing set is ',num2str(error_test_trans)]);
disp('//////////////////////////////////////////////////////////');

%% ------ Logistic regression with Binarized feature data -------

% % to add ones to first column of data --------------------
data_bin_logis = [ones(size(data_binary,1),1) data_binary];

% % add rows of ones ---------------------------
data_test_bin_logis = [ones(size(data_test_binary,1),1) data_test_binary];

% % to get lambda basing on minimum validation error ----------------
% lambda  = getlambda( data_bin_logis, data_training_labels )
lambda = 0.005; % i got lambda value to be :: lambda= 0.005 after minimizing the validation error

% Now we do training set error -------------------------
regs_param = findRegParam( data_bin_logis, data_training_labels, lambda );
error_train_trans = findError( data_bin_logis, data_training_labels, regs_param);
disp('///////////////////////////////////////////////////////////////////');
disp(['III. Error in training set using binarized data is ',num2str(error_train_trans)]);

% Here we do testing set error --------------------------
error_test_trans =  findError( data_test_bin_logis, data_testing_labels, regs_param);
disp(['III. Error in test set is ',num2str(error_test_trans)]);
disp('////////////////////////////////////////////////////////////////////');



%% %%%%%%%%%%%%//// HERE WE DO NAIVE BAYES METHOD /////%%%%%%%%%%%%%%%%
% We also have 'fitcnb' matlab inbuilt function to fit NB Classifier to the data
% but I implement it by using those functions' codes

%% Naive Byes using standardised preprocessing data-----------------

% To find the test labels of Naive classifier
test_lab_naive = getNaiveBayes( data_train_standize, data_training_labels, data_test_standize);

% To compute error of Testing data
error_stand_naive = findOnlyError( test_lab_naive, data_testing_labels );
disp('--------------------------------------------------');
disp('We get Naive Bayes Classifier Output');
disp(['Error in testing set using standardised data is ',num2str(error_stand_naive)]);

%% Naive Byes using transformed preprocessing data-----------------

% To get the test labels of Naive classifier
test_lab_naive = getNaiveBayes( data_train_transf, data_training_labels, data_test_transf );

% Computing Testing data error
error_trans_naive = findOnlyError( test_lab_naive, data_testing_labels );
disp('--------------------------------------------------');
disp('Naive Bayes Classifier Output::');
disp(['Error in testing set using transformed data is ',num2str(error_trans_naive)]);

%% Naive Byes using Binarized preprocessing data-----------------

% To get the testing labels of Naive classifier
test_lab_naive = getNaiveBayes( data_binary, data_training_labels, data_test_binary );

% Computing Testing data error
error_bin_naive = findOnlyError( test_lab_naive, data_testing_labels );
disp('----------------------------------------------------');
disp('Naive Bayes Classifier Result::');
disp(['Error in test set using binarized data is ',num2str(error_bin_naive)]);

