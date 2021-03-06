% MAIN.M FILE FOR FEATURE EXTRACTION
% RUN IT. YOU WILL GET FEATURES MFCC COEFF AND DCT MATRIX  
%%
files = dir('./tracks1/*.wav');

% here we input only 4 .wav files for now and then try replacing 4 with 729
% you will get the features for 729 music files
y = datasample(files,4,'Replace', false);
filename = cell(1, length(y));

% FEATURE SAVE IN FILE mfccReusults.xls or .mat 
save('./mfccResult/mfccResults.xls');
for k=1:length(y)
    file = strcat('./tracks1/',y(k).name);
    [~, name, ~] = fileparts(file);
    [mfcc_result,~] = mfcc_coeffs(file);
    varname = ['mfcc_' name];
    assignin('base', varname, mfcc_result);
    % FEATURE SAVE IN FILE mfccReusults.xls or .mat in all 4 .wav files 
    save('./mfccResult/mfccResults.xls', varname,'-append');
    clear mfcc_result file name varname
end