function [fc, ERB] = fc_ERB_hb(min_f, max_f, nChan)
% Generate the Q value plot as per [Acoustic flow...parameters] 
% Constants:
fr  = 4e4;  % foveal freq. [Hz] - Change to 40 later
q0  = 10;   % ration of maximum (qr=400) to minimum (q0=10) -10dB - filter quality
qr  = 400;

% Q -10dB values and ERB
% Parameters:
m   = [3.850547786150128e+18 3.051156234741211e+03];    % (f<=fr  f>fr)
f0  = [min_f max_f];

% f <= fr
q10_low     = @(fc)(qr-q0).*(m(1).^((f0(1)-fc)./(f0(1)-fr)) - 1)./(m(1)-1) + q0; %for fig
% f > fr
q10_high    = @(fc)(qr-q0).*(m(2).^((f0(2)-fc)./(f0(2)-fr)) - 1)./(m(2)-1) + q0; %for fig

% Calculate Step Factor
% Formulas: Q=fc/BW=fc/ERB, now, i/ERB=1/fc*Q. and step=(integration of i/ERB)/no of channel

% factor to adjust for 10dB bandwidth (0.5) -> 3 dB bandwidth -> ERB  (0.887):
kappa   = 2*.887;
% f <= fr
invERB_1    = @(fc)(kappa./fc).*((qr-q0).*(m(1).^((f0(1)-fc)./(f0(1)-fr)) - 1)./(m(1)-1) + q0);
% f > fr
invERB_2    = @(fc)(kappa./fc).*((qr-q0).*(m(2).^((f0(2)-fc)./(f0(2)-fr)) - 1)./(m(2)-1) + q0);

step_factor = (integral(invERB_1,f0(1),fr)+ integral(invERB_2,fr,f0(2)))/nChan;

% Calculate the actual centre Frequency fc
% Find center frequency for each channel index nummerically:
LMAX    = 100;              % max number of iterations:
i_ch    = 0:nChan-1;
fc      = zeros(nChan,1);   % preallocate center freqs

for i = 1:nChan
  fc(i) = f0(1);        % start bisecting at f0_l
  df    = f0(2) - f0(1);
  for l = 1:LMAX
    df      = df/2;         %half of df 
    lb_mid  = fc(i) + df;
    if lb_mid > fr
        G_mid   = integral(invERB_2,lb_mid, f0(2))./step_factor - i_ch(i);
    end
    if lb_mid < fr
        G_mid   = ((integral(invERB_2,fr, f0(2)) + integral(invERB_1,lb_mid, fr))./step_factor) - i_ch(i);
    end
    if (G_mid>0)
        fc(i)   = lb_mid;
    end
    if abs(df)<=eps
        break
    end
  end
end
fc  = flipud(fc);
fc  = fc';

% split fc in fc <= fr & fc > fr:
idx_l   = find(fc <= fr);
idx_h   = find(fc >  fr);

%fc <= fr
invERB_1    = (kappa./fc(idx_l)).*((qr-q0).*(m(1).^((f0(1)-fc(idx_l))./(f0(1)-fr)) - 1)./(m(1)-1) + q0);
% f > fr
invERB_2    = (kappa./fc(idx_h)).*((qr-q0).*(m(2).^((f0(2)-fc(idx_h))./(f0(2)-fr)) - 1)./(m(2)-1) + q0);
%invERB2 placed first because the number of channels are assingned from 3000 to 1 (not 1 to 3000) in the paper
ERB     = 1./[invERB_1 invERB_2];
end