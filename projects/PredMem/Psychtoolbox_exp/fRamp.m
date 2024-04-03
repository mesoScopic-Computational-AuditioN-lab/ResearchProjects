% this script adds a 10 ms ramp-up and ramp-down to the
%stimulus. 
%The Sampling frequency is hard-coded as 44,000 Hz.

function stim=framp(sz,sig,fs)

if nargin <3
    fs= 44100;
end

stim = sig;

len = length(stim);
%time = len/fs; %length in seconds

%computing how many points represent sz ms
points = sz*1/1000*fs;
points=round(points);

%fitting the points with a cosine function
freq = fs/(2*points);
sd = 1/fs;
n=1:points;
t=n*sd;
ramp1=((1+(cos(2*pi*freq*t + pi)))/2).^2;
ramp2= ((1+(cos(2*pi*freq*t )))/2).^2;
%creating the filter

ramp = [ramp1 ones(1,len-length(ramp1) - length(ramp2)) ramp2];

a=stim;

stim = ramp.*a;
end