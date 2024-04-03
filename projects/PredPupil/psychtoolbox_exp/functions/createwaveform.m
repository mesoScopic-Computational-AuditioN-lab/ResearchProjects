function xt = createwaveform(freq, stim_len, samp_rate, phase)
% Generate sound stimulus for some length 

if nargin<4     % set phase to 0 if none was given
    phase = 0;
end
phase   = (phase/360)*samp_rate; % compute phase in degrees ralative to samp_rate
xt      = sin(2*pi*freq'*(0-phase:(stim_len*samp_rate)-phase-1)/samp_rate); % generate waveform

return;