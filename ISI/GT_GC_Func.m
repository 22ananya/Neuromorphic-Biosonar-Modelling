function [GT_GC_out] = GT_GC_Func(model, t, fs, nChan, fc, ERB, coef_b, coef_c, data)
imp     = zeros(length(t),nChan);
n       = 4;
tpt     = (2*pi)/fs;
gain    = (((coef_b^2).*ERB.*tpt).^n)./6;  %%See fast gammatone ()

for i = 1:nChan
    env     = gain(i)*(fs^3)*t.^(n-1).* exp(-2*pi*coef_b*ERB(i)*t);
    if model == 1        
        carrier     = cos(2*pi*fc(i)*t);
    elseif model == 2
        carrier     = cos(2*pi*fc(i)*t+coef_c*log(t));
    end
    imp(:,i)    = env.*carrier;
end
GT_GC_out = fftfilt(imp,repmat(data,1,nChan));