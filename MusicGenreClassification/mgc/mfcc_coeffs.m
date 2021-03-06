function [mfcc, DCT] = mfcc_coeffs(filename)
% this function is used to extract mfc-coefficients 
[wav, fs] = audioread(filename);
p.fs = fs;
p.visu = 0;
p.hopsize = 128;
[mfcc, DCT] = ma_mfcc(wav, p);
end