% Make dct basis test files using SPM
% Needs SPM12 on the path
dct_5 = spm_dctmtx(5);
save dct_5.txt -ascii dct_5;
dct_10 = spm_dctmtx(10);
save dct_10.txt -ascii dct_10;
dct_100 = spm_dctmtx(100);
save dct_100.txt -ascii dct_100;
