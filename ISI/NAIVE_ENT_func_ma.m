function [TE, lb, w] = NAIVE_ENT_func_ma(train, dt, T)
%Entropy using the basic formula sum(-pi*log2(pi)) this is Called Direct
%Entropy Naive Estimate
%Spike Train is the input to this function

wordlength  = T/dt;                 % # of sampled units of spike train in one word
sizet       = length(train);        % Adjust depending on whether input is an array or a vector
nwords      = round(sizet/wordlength);   % Total number of words = round(sizet/T);

%Divide Spike train into words
w       = zeros(nwords,wordlength);
sizew   = nwords*wordlength;

for i = 1:nwords-1
    if (i*wordlength)<=sizet
        w(i,:) = train(1+((i-1)*wordlength):1:(i*wordlength));
    else
        w(i,1:wordlength-(sizew-sizet)) = train(1+((i-1)*wordlength):1:sizet);
    end
end

%count unique words
wordcount   = zeros(size(w,1),1);
for a = 1:size(w,1)
    tempword = w(a,:);
    for b = 1:size(w,1)
        if w(b,:) == tempword
            wordcount(b,1) =  wordcount(b,1) + 1;
        end
    end
end

wset    = [w wordcount];
%create an additional array of words and occurences and remove duplicates
wset    = unique(wset, 'rows');

%to create a count of number of words for given number of spikes, simple
%addition can give that

%also count differently to validate
nwords2 = sum(wset(:,end));

%create probability array
pi  = wset(:,end)./nwords2;
pc = sum(pi.^2); %lower bound of entropy



entropy = 0;
for i = 1:length(pi)
    entropy = entropy -(pi(i)*log2(pi(i)));
end
TE      = sum(entropy);
lb = -log2(pc);
end