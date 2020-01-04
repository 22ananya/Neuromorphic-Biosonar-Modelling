function [IHC_out] = IHC_BOX(BM, fs, nChan)
for ch = 1:nChan
    y   = zeros(1,length(BM(:,ch)));
    for i = 1:length(BM(:,ch))
        y(i)    = BM(i,ch)*(BM(i,ch)>0);
    end
    % Low Pass Filter
    cutoff  = 1000;
    [f1, f2] = butter(2, cutoff*2/fs);
    IHC_out(:,ch) = filter(f1, f2, y);
end