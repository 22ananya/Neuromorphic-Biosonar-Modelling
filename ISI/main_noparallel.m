%% Full_like main 4
%% Main

 clear all
 close all
% 
% %% Loading Data Set
clear all
%clearvars -except full_sta full_dyn
close all
load('full_data.mat') %TODO for Full data
%   load('10_12_CRC_noc_sta_4.mat')
%  full_sta = data_mat;
%  load('10_12_CRC_noc_dyn_4.mat')
%  full_dyn = data_mat(:,1:size(full_sta,2));

data_range = 'STW5';
BM_model = 3;
SR_model = 1;
for q = 1:2
    if q == 1
        file = full_sta;
        count = 1;
    else
        file = full_dyn;
        count = 2;
    end
    %% Center Frequencies Setting
    min_f       = 20e3;     % Minimum Frequency
    max_f       = 45e3;     % Maximum Frequency
    nChan       = 100;       % Number of channels
    fs          = 400e3;    % Sampling rate Fs in Hz
    [fc, ERB]   = fc_ERB_hb(min_f, max_f, nChan); 

    %nData       = length(data_range); TODO                                        % Data set
    %data_mat    = double(file(6001:10000,data_range)); %zeros(3000,nData)]; %ones(3000,nData).*file(10000,:)];    % One-ear data set
    data_mat    = double(file(6001:10000, 6439:3:8035));
    nData       = size(data_mat,2); % if loading specific data file
    N           = 7000;                            % Number of samples TODO
    t           = 1/fs:1/fs:N/fs;                               % Time axis

    %% Models' Parameters
    % BM's Parameters
    leveldBSPL  = 50;
    coef_b      = 1.019;    % Increasing or decreasing of b means increasing or decreasing ERB
    coef_c      = -3;       % Controls the degree of asymmetry in gammachirp, when c=0, gammachirp=gammatone
    coef_GTGC   = [coef_b, coef_c];
    coef_d      = 3.1; %4.2040 3.1;   % Linear gain coef 1
    coef_e      = -0.6; %-0.4791;  % Linear gain coef 2
    coef_f      = 1.17274;  % Non-Linear bandwidth broken stick nonlinearity coef 1
    coef_g      = 0.0113;   % Non-Linear bandwidth broken stick nonlinearity coef 2
    coef_h      = 1.4030;   % a1 Broken-stick nonlinerity cofficients
    coef_i      = 0.8192;   % a2 Broken-stick nonlinerity cofficients
    coef_j      = 1.6191;   % b1 Broken-stick nonlinerity cofficients
    coef_k      = -0.8187;  % b2 Broken-stick nonlinerity cofficients
    coef_l      = -0.6021;  % c1 Broken-stick nonlinerity cofficients
    coef_m      = 0;        % c2 Broken-stick nonlinerity cofficients
    coef_DRNL   = [coef_d, coef_e, coef_f, coef_g, coef_h, coef_i, coef_j, coef_k, coef_l, coef_m];
    BM_par      = [N, fs, nChan, leveldBSPL, coef_GTGC, coef_DRNL];

    % SRM's Parameters
    V_rest      = 0;        % resting membrane potential [V]
    V_reset     = 0;    % after-potential hyperpolarization [V]
    VarTheta    = 9e-5;     % spike threshold [V]
    G_m         = 1.67e+2;  % membrane conductance [S] (MAP parameter human)
    Tau_m       = 0.0075;    % membrane time constant [s] (MAP parameter for human)
    SRM_par     = [N, fs, nChan, V_rest, V_reset, VarTheta, G_m, Tau_m];
    
    % Entropy's Parameters
    dt      = 0.5e-3;         % Time bin for spike train entropy calc %TODO
    T       = 14e-3;        % Word length for spike train entropy calc TODO if adding zeros 14
    ENT_par = [N, fs, nData, dt, T];

    %% Model
    for i = 1:nData %parfor here
        data        = commonbandpass(min_f, data_mat(:,i), max_f, fs);  % Low Pass Filter
        data        = data - mean(data); % Zero mean correction
        %size(data)
        data = [data ; zeros(3000,1)]; %comment out to revert
        %size(data)
        normalizer  = sqrt(sum(data.^2));
        data        = data./normalizer;
        
    % Basilar Membrane Displacement Function
        [BM_out(:,:,i)]    = BM_BOX(BM_model, data, BM_par, fc, ERB);  % 1: GT, 2: GC, 3: DRNL
    %normalize BM_out    
         %BM_out(:,:,i)      = BM_out(:,:,i) - mean(BM_out(:,:,i)); % Zero mean correction
         temp = max(max(abs(BM_out(:,:,i))));
%          temp = BM_out(:,:,i);  %Uncomment from here
%          for j = 1:nChan
%              normalizer  =        sqrt(sum(temp(:,j).^2));
%              temp(:,j)       = temp(:,j)./normalizer;
% %              normalizer  =        max(abs(temp(:,j)));
% %              temp(:,j)       = temp(:,j)./normalizer;
%              
%          end
%          BM_out(:,:,i) = temp; % To here to revert to old full bm
%          normalization and comment out next line
         BM_out(:,:,i) = BM_out(:,:,i)./temp;
        % Inner Hair Cell Calcium Current Function
        [IHC_out(:,:,i)]   = IHC_BOX(BM_out(:,:,i), fs, nChan);   
        
        % Spiking Activity Function
        [SRM_out(:,:,i), Tspikes(:,i)]  = SRM_BOX(SR_model,  IHC_out(:,:,i), SRM_par);     % 1: Leaky IAF, 2: Kernels
    end
 fc = fc;
    %% Entropy 
    %Complete Entropy Calculations - Both Direct and CDM

   
    %% Clear stuff you don't need
    clearvars -except SRM_out nData nChan TE lb TE_CDM x Entropy count full_dyn TE_neuron BM_out IHC_out IHC1 BM1 fc BM_model SR_model data_range w1 VarTheta Tau_m N

    %% add various paths needed

    %p = mfilename('fullpath');
    %[~, idx] = find(p == '/', 1, 'last');
    %base = p(1:idx);
    base = '/home/ananya22/CDMentropy-master/';
    % add libraries
    addpath([base 'lib/PYMentropy/src/']);

    % add CDM
    addpath([base 'src/']);

    %% Define required parameters for Entropy Calculations
    ENT_par = [N, 400000, nChan, nData, 0.0005, 0.014]; %length of echo, sampling rate, nChan, nData, dt, T %TODO First value = N to 7000
    s1 = ENT_par(4)*(ENT_par(6)/ENT_par(5));
    Train_Mat2 = zeros(s1,ENT_par(3));
    [TE(count),Entropy(count,:),lb(count,:),w,Train_Mat2] = ENT_BOX(SRM_out, ENT_par);	

    %% Create figure - Direct Method
    figure;
    bar(Entropy(count,:))
    hold on
    bar(lb(count,:),'r')
    xlabel('Channel no.')
    ylabel('Entropy(bits)')
    legend('Entropy','Lower bound')
    title('Entropy vs lower ma bound')

    %% Do CDM Method analysis
%     x = zeros(2,nChan);
%     for j = 1:nChan
%         x(count,j) = x(count,j) + computeH_CDM(w(:,:,j));
%     end
%     TE_CDM(count) = sum(x(count,:));
    
    %% Do CDM Method analysis
%     x2 = zeros(2,1);
%     for k = 1:2
%         x2(count) = x2(count) + computeH_CDM(Train_Mat2(:,1+(k-1)*50:k*50));
%     end
%     TE_neuron(count) = sum(x2(count));
    
    if count == 1
        w1 = w;
        SRM1 = SRM_out;
        %IHC1 = IHC_out;
        %BM1 = BM_out;
    else
        w2 = w;
        SRM2 = SRM_out;
        %IHC2 = IHC_out;
        %BM2 = BM_out;
    end   
end
c = clock
file_name = datestr(c,0)

save(file_name, 'BM_model', 'SR_model', 'TE_CDM' , 'w1' , 'w2', 'data_range')
    
for i = 1:nChan; s1(i) = sum(sum(w1(:,:,i)));end
s1 = s1./nData;
c = {'Static','Dynamic'};
figure;bar(s1);title('Spike count - Static');xlabel('Channel no.');ylabel('Number of spikes');xlim([0 nChan]);

for i = 1:nChan; s(i) = sum(sum(w2(:,:,i)));end
s = s./nData;
c = {'Static','Dynamic'};
figure;bar(s);title('Spike count - Dynamic');xlabel('Channel no.');ylabel('Number of spikes');xlim([0 nChan]);

temp = s-s1;
figure;bar(temp);title('Spike count - Difference D-S');xlabel('Channel no.');ylabel('Number of spikes');xlim([0 nChan]);

figure;bar(Entropy(1,:));hold on;bar(Entropy(2,:),'r');legend('Static','Dynamic');xlim([0 nChan]);
figure;bar(TE);title('Total Entropy'),ylabel('Entropy (bits)');set(gca,'xticklabel',c)
figure;bar(Entropy(2,:) - Entropy(1,:),'r');title('Dynamic - Static Entropy');xlabel('Channel no.');ylabel('Entropy (bits)');xlim([0 nChan]);    