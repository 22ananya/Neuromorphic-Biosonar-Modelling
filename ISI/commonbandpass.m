function R=commonbandpass(lowfreq,RX,highfreq,fs)
order=10; % order of the filter
Wn2=(lowfreq)/(fs/2);
[b2, a2]=butter(order,Wn2,'high'); 
filterR=filtfilt(b2,a2,RX);
Wn2=(highfreq)/(fs/2);
[b2, a2]=butter(order,Wn2,'low'); 
R=filtfilt(b2,a2,filterR);
end