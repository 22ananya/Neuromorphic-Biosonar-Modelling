function [SpMat, V_plot, Nspikes, T] = SRM_K_func(SRM_par, I)
% Parameters
N           = SRM_par(1);
fs          = SRM_par(2);
dt          = 1/fs;             % time step [s]
t_end       = N/fs;             % signal time length [s]
V_rest      = SRM_par(4);
V_reset     = SRM_par(5);
VarTheta    = SRM_par(6);
R_m         = (1/SRM_par(7));   % membrane resistance (1/G_m) [Ohm];
Tau_m       = SRM_par(8);
Tau_refr    = Tau_m;            % refractory time constant [s]
Tau_rec     = Tau_m;            % recovery time constant [s]
V_spike     = VarTheta+abs(VarTheta-V_reset);
Eta_0       = V_rest-V_reset;

% Initial Conditions
i           = 1;
V           = zeros(1,N);
V_plot      = V;
SpMat       = V;
V(i)        = V_rest;
V_plot(i)   = V_rest;
K_Eta       = V';
K_Kap       = zeros(N,1);
Thres       = VarTheta*ones(N,1);

% Integration: V(t) = Eta(t-ti) + int[Kappa(s).I(t-s)ds]
Nspikes     = 0;
ti(1,Nspikes+1)   = 0;
for k = dt:dt:t_end-dt
    t = k-ti(1,Nspikes+1);

    if t >= 0 && ti(1,Nspikes+1) ~= 0
        K_Eta(i)    = V_rest-Eta_0*exp(-t/Tau_refr);
        Thres(i)    = VarTheta+(VarTheta-V_reset)*exp(-t/Tau_refr);
    else
        K_Eta(i)    = V_rest;
    end

    if t > 0
        K_Kap(i)    = (R_m/Tau_m)*(1-exp(-t/Tau_rec))*I(i)*Tau_m*(1+exp(-t/Tau_m));
    end
    
    V(i)    = K_Eta(i) + K_Kap(i);
    
    if V(i) > Thres(i)
        V(i)        = V_reset;
        V_plot(i)   = V_spike;
        SpMat(i+1)  = 1;
        Nspikes     = Nspikes+1;
        ti(1,Nspikes+1)   = k;
    else
        V_plot(i)   = V(i);
    end    
    i = i+1;
end

if Nspikes == 0
    T = 0;
else
    T = Nspikes/t_end;
end
end