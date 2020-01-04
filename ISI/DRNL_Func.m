function [DRNL_out] = DRNL_Func(fs, nChan, fc, ERB, coef_d, coef_e, coef_f, coef_g, coef_h, coef_i, coef_j, coef_k, coef_l, coef_m, data)
for i=1:nChan
    lin_fc          = fc(i);
    lin_bw          = ERB(i); 
    lin_gain        = 10.^(coef_d+coef_e*log10(fc(i)));
    lin_lp_cutoff   = lin_fc;
    nlin_fc_before  = lin_fc./1;    % 0.97%In Brian toolbox it same as lin_fc
    nlin_fc_after   = nlin_fc_before;
    nlin_bw_before  = lin_bw./(coef_f.*fc(i).^coef_g); %bandwidth of nonlinear part before broken stick nonlinearity. Note nonlinear bandwidth is always lower than linear bandwidth. defaults value is [1.17274  0.0113]
    nlin_bw_after   = nlin_bw_before;
    nlin_lp_cutoff  = nlin_fc_before;
        
    % Broken-stick nonlinearity coefficients: a, b, c
    nlin_a  = 10.^(coef_h+coef_i*log10(fc(i)));
    nlin_b  = 10.^(coef_j+coef_k*log10(fc(i)));
    nlin_c  = 10.^(coef_l+coef_m*log10(fc(i)));     %the results is 0.25 which satisfy Brian's Toolbox and others

    % Compute gammatone coefficients for the linear stage
    [GTlin_b,GTlin_a]   = FOD_GF(lin_fc, 2, fs, lin_bw);

    % Compute coefficients for the linear stage lowpass, use 2nd order Butterworth.
    [LPlin_b,LPlin_a]   = butter(2, lin_lp_cutoff/(fs/2));

    % Compute gammatone coefficients for the non-linear stage
    [GTnlin_b_before, GTnlin_a_before]  = FOD_GF(nlin_fc_before, 3, fs, nlin_bw_before);
    [GTnlin_b_after,GTnlin_a_after]     = FOD_GF(nlin_fc_after, 3, fs, nlin_bw_after);

    % Compute coefficients for the non-linear stage lowpass, use 2nd order Butterworth.
    [LPnlin_b,LPnlin_a] = butter(2, nlin_lp_cutoff/(fs/2));

    % -------------- linear part ----------------
    y_lin   = data.*lin_gain;
    
    % Gammatone filtering
    y_lin   = filter(GTlin_b, GTlin_a, y_lin);

    % Multiple LP filtering
    lin_nlp = 4;
    
    for j = 1:lin_nlp
        y_lin   = filter(LPlin_b, LPlin_a, y_lin);
    end
    
    % ------------ non-linear part --------------
    y_nlin  = filter(GTnlin_b_before, GTnlin_a_before, data);

    % Broken stick nonlinearity
    y_nlin  = sign(y_nlin).*min(nlin_a*abs(y_nlin),nlin_b*(abs(y_nlin)).^nlin_c);
    y_nlin  = filter(GTnlin_b_after,GTnlin_a_after,y_nlin);

    % Then LP filtering
    nlin_nlp    = 3; 
    for j = 1:nlin_nlp
        y_nlin = filter(LPnlin_b,LPnlin_a,y_nlin);
    end
    DRNL_out(:,i)   = y_lin + y_nlin;
end
end