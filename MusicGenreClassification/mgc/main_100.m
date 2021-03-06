files = dir('./tracks1/*.wav');
% samples of 100 files
ybar = datasample(files,729,'Replace', false);
filename = cell(1, length(ybar));

save('./mfccResult/mfccResults_729.mat');
for k=1:length(ybar)
    file = strcat('./tracks1/',ybar(k).name);
    [~, name, ~] = fileparts(file);
    [mfcc_result,~] = mfcc_coeffs(file);
    varname = ['mfcc_' name];
    assignin('base', varname, mfcc_result);
    save('./mfccResult/mfccResults_100.mat', varname,'-append');
    clear mfcc_result file name varname
end