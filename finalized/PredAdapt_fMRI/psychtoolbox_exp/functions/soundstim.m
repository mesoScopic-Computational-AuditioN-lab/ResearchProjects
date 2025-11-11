function xt = soundstim(freq, stim_len, samp_rate)
% Generate sound stimulus for some length

xt = sin(2*pi*freq*(0:stim_len*samp_rate-1)/samp_rate);

return;