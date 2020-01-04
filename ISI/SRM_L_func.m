function [SpMat, V_plot, Nspikes, T] = SRM_L_func(SRM_par, I)
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
V_spike     = VarTheta+abs(VarTheta-V_reset);

% Initial Conditions
i           = 1;
V           = zeros(1,N);
V_plot      = V;
SpMat       = V;
V(i)        = V_rest;
V_plot(i)   = V_rest;

% Integration: Tau*dV/dt = V_rest - V(t) + R*I(t)
Nspikes     = 0;
for k = dt:dt:t_end-dt
    V_inf   = V_rest + R_m*I(i);
    V(i+1)  = V_inf+(V(i)-V_inf)*exp(-dt/Tau_m);
    
    if V(i+1) > VarTheta
        V(i+1)      = V_reset;
        V_plot(i+1) = V_spike;
        SpMat(i+1)  = 1;
        Nspikes     = Nspikes+1;
    else
        V_plot(i+1) = V(i+1);
    end  
    i = i+1;
end

if Nspikes == 0
    T = 0;
else
    T = Nspikes/t_end;
end
end