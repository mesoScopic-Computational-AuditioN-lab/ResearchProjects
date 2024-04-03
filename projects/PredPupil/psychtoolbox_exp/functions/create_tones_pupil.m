function [raw_tones, mod_tones] = create_tones_pupil(stimuli, stim_len, samp_rate, rampup_dur, rampdown_dur)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                    %
%   CREATE TONES FOR TONOTOPY (INC. AMP MODULATION AND RAMPUP/DOWN   %
%                                                                    %
%   Input:                                                           %
%                                                                    %
%   - stimuli :         frequencie array (sequentialy ordered) in hz %
%   - stim_len :        a set stimulus length in seconds             %
%   - samp_rate :       sampling rate used                           %
%                                                                    %
%   - rampup_dur :      rampup duration to be used                   %
%   - rampdown_dur :    rampdown duration to be used                 %
%                                                                    %
%   Outputs:    returns both raw and modulation tones, that are      %
%               also saved.                                          %
%                                                                    %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Set function handles and precreate arrays

% set function handles for stimulus and modulation
stim    = @(p) [createwaveform(p, stim_len, samp_rate)];

% rampup array
rampup_samps = floor(samp_rate * rampup_dur);
w_up = hann(2*rampup_samps)';            % hanning / cosine^2 ramp
w_up = w_up(1:ceil((length(w_up))/2));
rampdown_samps = floor(samp_rate * rampdown_dur);
w_down = hanning(2*rampdown_samps)';     % hanning / cosine^2 ramp
w_down = w_down(ceil((length(w_down))/2)+1:end);
w_on_xt = [w_up ones(1,(stim_len*samp_rate)-length(w_up))]; % get on and off in array
w_off_xt = [ones(1,(stim_len*samp_rate)-length(w_down)) w_down];

% define return matrix
stimulilength = size(stim(stimuli{1}), 2);
segmentlength = cellfun(@(c) size(c, 2), stimuli);
raw_tones = arrayfun(@(x) nan(x, stimulilength), segmentlength, 'UniformOutput', false);   % tones whitout any modulation
mod_tones = arrayfun(@(x) nan(x, stimulilength), segmentlength, 'UniformOutput', false);   % tones with rampup/down

%% Main loop
for s = 1:size(stimuli,2)

    % generate tones
    cur_tones               = stim(stimuli{s});                   % get current tones
    raw_tones{s}            = cur_tones;                          % put into matrix
    mod_tones{s}            = cur_tones.*w_on_xt.*w_off_xt;       % apply ramp

end

return