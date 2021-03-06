% This is the main file of the GTZAN Project
% Path setting and Tools for the for the project; Toolboxes mentioned in
% the report
%%
addpath 'C:\Users\Vamshi\Downloads\try 2\utility' 
addpath 'C:\Users\Vamshi\Downloads\try 2\sap'
addpath 'C:\Users\Vamshi\Downloads\try 2\machineLearning'
addpath 'C:\Users\Vamshi\Downloads\try 2\machineLearning\externalTool\libsvm-3.21\matlab'

%% user comments displayed on the command line
fprintf('Platform: %s\n', computer);
fprintf('MATLAB version: %s\n', version);
fprintf('Script starts at %s\n', char(datetime));
scriptStartTime=tic;	% Timing for the whole script

%% Input the music files
auDir='C:\Users\Vamshi\Downloads\try 2\dataset\genres';
opt=mmDataCollect('defaultOpt');            % struct with 5 fields
opt.extName='au';
auSet=mmDataCollect(auDir, opt, 1);

%% Feature Extraction
if ~exist('ds.mat', 'file')
	myTic=tic;
	opt=dsCreateFromMm('defaultOpt');
	opt.auFeaFcn=@mgcFeaExtract;	% Function for feature extraction
	opt.auFeaOpt=feval(opt.auFeaFcn, 'defaultOpt');	% Feature options
	opt.auEpdFcn='';		% No need to do endpoint detection
	ds=dsCreateFromMm(auSet, opt, 1);
	fprintf('Time for feature extraction over %d files = %g sec\n', length(auSet), toc(myTic));
	fprintf('Saving ds.mat...\n');
	save ds ds
else
	fprintf('Loading ds.mat...\n');
	load ds.mat
end

% files with .au extension
auFile=[auDir, '\disco\disco.00001.au'];

% MFCC has 39 dimensions, the extracted file-based features has 156 (= 39*4) dime
figure; mgcFeaExtract(auFile, [], 1);

% Data Visualization -----
figure;
[classSize, classLabel]=dsClassSize(ds, 1);

figure; dsRangePlot(ds);

figure; dsFeaVecPlot(ds); figEnlarge;

dim=size(ds.input, 1);
fprintf('Feature dimensions = %d\n', dim);

%% Applying PCA for the Dimension Reduction
[input2, eigVec, eigValue]=pca(ds.input);
cumVar=cumsum(eigValue);                    % 156X1 double
cumVarPercent=cumVar/cumVar(end)*100;
figure; plot(cumVarPercent, '.-');
xlabel('No. of eigenvalues');
ylabel('Cumulated variance percentage (%)');
title('Variance percentage vs. no. of eigenvalues');

cumVarTh=95;
index=find(cumVarPercent>cumVarTh);         % 147X1 double
newDim=index(1);                    
ds2=ds;
ds2.input=input2(1:newDim, :);
fprintf('Reduce the dimensionality to %d to keep %g%% cumulative variance via PCA.\n', newDim, cumVarTh);

%% LDA ----
ds2d=lda(ds);                               % struct with 6 fields
ds2d.input=ds2d.input(1:2, :);
figure; dsScatterPlot(ds2d); xlabel('Input 1'); ylabel('Input 2');
title('Features projected on the first 2 lda vectors');

%% K-NN classifier for the music classification
[rr, ~]=knncLoo(ds);
fprintf('rr=%g%% for original ds\n', rr*100);
ds2=ds; ds2.input=inputNormalize(ds2.input);   % struct with 6 fields
[rr2, computed]=knncLoo(ds2);
fprintf('rr=%g%% for ds after input normalization\n', rr2*100);

myTic=tic;
mgcOpt=mgcOptSet;
if mgcOpt.useInputNormalize, ds.input=inputNormalize(ds.input);	end		% Input normalization
cvPrm=crossValidate('defaultOpt');                                      % struct with 3 fields
cvPrm.foldNum=mgcOpt.foldNum;
cvPrm.classifier=mgcOpt.classifier;
plotOpt=1;
figure
% performing cross validation----------
[tRrMean, vRrMean, tRr, vRr, computedClass]=crossValidate(ds, cvPrm, plotOpt); figEnlarge;
fprintf('Time for cross-validation = %g sec\n', toc(myTic));

% Finding Confusion Matrix--------
for i=1:length(computedClass)
	computed(i)=computedClass{i};
end
desired=ds.output;                              % 1X1000 double
confMat = confMatGet(desired, computed);        % 10X10 double
cmOpt=confMatPlot('defaultOpt');
cmOpt.className=ds.outputName;                  % 1X10 classNames array
confMatPlot(confMat, cmOpt); figEnlarge;        % Plot the confusion matrix