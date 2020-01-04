function [SRM_out, Tspikes] = SRM_BOX(model, IHC_out, SRM_par)
% Parameters
N           = SRM_par(1);
nChan       = SRM_par(3);
SRM_out     = zeros(N, nChan);
V           = zeros(N, nChan);
Nspikes     = zeros(nChan,1);
Tspikes     = zeros(nChan,1);

if model == 1
    % Leaky Integrate-And-Fire
    for ch = 1:nChan
        [SRM_out(:,ch), V(:,ch), Nspikes(ch), Tspikes(ch)] = SRM_L_func(SRM_par, IHC_out(:,ch));
    end
elseif model == 2
    % SRM responses Kernel
    for ch = 1:nChan
        [SRM_out(:,ch), V(:,ch), Nspikes(ch), Tspikes(ch)] = SRM_K_func(SRM_par, IHC_out(:,ch));
    end
end

%{
for ch = 1:nChan
    figure(ch);
    subplot(211)
    plot(t,V(:,ch))
    str=[num2str(Nspikes(ch)),' spike(s)'];
    text(t(end)/2.5,(7/8)*(max(V(:,ch))+min(V(ch,:))),str);
    title(['Leaky Integrate-And-Fire, channel #',num2str(ch),', ',num2str(fc(ch)/1000),' kHz'])
    xlabel('Time [s]')
    ylabel({'Membrane','Potential','[V]'})
    subplot(212)
    plot(t,IHC_out(:,ch))
    title(['Calcium current form the IHC, channel #',num2str(ch),', ',num2str(fc(ch)/1000),' kHz'])
    xlabel('Time [s]')
    ylabel({'Current','[A]'})
end
%}
end