function [TE,Entropy, lb,w,Train_Mat2] = ENT_BOX(SRM_out, ENT_par)
%Entropy Function:
%The input Spike Matrix has to have all 500 (number of stimuli) spike train
%sets for all nc neurons
%Create a loop or pull out info such that you're looking at all instances
%of spike trains of a specific neuron

% Parameters
N       = ENT_par(1);
fs      = ENT_par(2);
nChan   = ENT_par(3);
nData   = ENT_par(4);
dt      = ENT_par(5);
T       = ENT_par(6);

nbin        = fs*dt; %No. of samples in each bin as each bin is 1ms long
Train_Mat   = zeros(round(N/nbin),nChan,nData);

%bichanneling 
for k = 1:nData
    for j = 1:nChan
        for i = 1:nbin:N-nbin
            Train_Mat(1+(i-1)/nbin,j,k)   = sum(SRM_out(i:i+nbin-1,j,k));
            if Train_Mat(1+(i-1)/nbin,j,k)>=1
                Train_Mat(1+(i-1)/nbin,j,k)   = 1;
            else
                Train_Mat(1+(i-1)/nbin,j,k)   = 0;
            end
        end
    end
end

%ISI OF one specific channel
%for i = 1:nChan - a loop would only work if you have same spikes for each
%channel

% full_ST = zeros(N, nChan*nData);
Train_Mat2 = [];

%Make One Long Spike Train out of all of the spike trains for the same neuron
for i = 1:nData
    Train_Mat2 = [Train_Mat2;Train_Mat(:,:,i)];
end

%calculating entropy neuron by neuron
Entropy = zeros(1,nChan);
% H_LB = zeros(1,nChan);
% H_UB = zeros(1,nChan);

%w = zeros(nData,28,nChan);

parfor i = 1:nChan
    [Entropy(1,i), lb(1,i),w(:,:,i)] = NAIVE_ENT_func_ma(Train_Mat2(:,i),dt,T);
%     [H_LB(1,i), H_UB(1,i)]  = TRUE_ENT_func(Train_Mat2(:,i),dt,T);
end

%Total Entropy in Bits
%sum of entropy in all neurons
TE      = sum(Entropy);
% TH_LB   = sum(H_LB);
% TH_UB   = sum(H_UB);
end