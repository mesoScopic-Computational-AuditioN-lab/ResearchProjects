function sig=fgenTone_adjust(freq,seg_size,fs)

if nargin <3
    fs= 44100;
end

%create a tone of frequency f
sd = 1/fs;
n=1:fs*(seg_size)/1000;
%N=length(n);
t=n*sd;
stim=sin(2*pi*freq*t);

%%add scaling factor (for equal loudness)
sfact=fgetLoudnessCorrectionFactor(freq);

stim=sfact*stim;
full_stim=fRamp(5,stim,fs); % add 5ms ramp-up and ramp-down to the stimulus
sig=full_stim;
end