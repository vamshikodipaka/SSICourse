% Function to extract mfcc coefficients and DCT
%% 
function [mfcc, DCT] = mfcc_coeffs(filename)
% Read files
[wav, fs] = audioread(filename);
% Draw features from the file with specifications
p.fs = fs;
p.visu = 0;
p.hopsize = 128;
% Extract feature components 
[mfcc, DCT] = ma_mfcc(wav, p);
end