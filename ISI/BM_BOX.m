function [BM_out] = BM_BOX(model, data, BM_par, fc, ERB)
% Parameters
N           = BM_par(1);
fs          = BM_par(2);
nChan       = BM_par(3);
leveldBSPL  = BM_par(4);
coef_b      = BM_par(5);
coef_c      = BM_par(6);
coef_d      = BM_par(7);
coef_e      = BM_par(8);
coef_f      = BM_par(9);
coef_g      = BM_par(10);
coef_h      = BM_par(11);
coef_i      = BM_par(12);
coef_j      = BM_par(13);
coef_k      = BM_par(14);
coef_l      = BM_par(15);
coef_m      = BM_par(16);

BM_out      = zeros(N, nChan);   % Empty BM Output Array
t           = 1/fs:1/fs:N/fs;    % Time axis

if model == 1 || model == 2
    % Gammatone / Gammachirp
    data            = leveldBSPL*data*0.00014*20;   %%20 has been multiplied to make the amp same as DRBL    
    [GT_GC_out]     = GT_GC_Func(model, t, fs, nChan, fc, ERB, coef_b, coef_c, data);
    BM_out          = GT_GC_out;    
elseif model == 3
    % DRNL
    data        = leveldBSPL*data*0.00014;
    [DRNL_out]  = DRNL_Func(fs, nChan, fc', ERB', coef_d, coef_e, coef_f, coef_g, coef_h, coef_i, coef_j, coef_k, coef_l, coef_m, data); 
    BM_out      = DRNL_out;
end
end