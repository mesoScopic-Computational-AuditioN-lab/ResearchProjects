function [raw_tones, mod_tones] = create_tones_main(stimuli, stim_len, samp_rate, rampup_dur, rampdown_dur)

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
raw_tones = nan(size(stimuli,1), size(stimuli,2),  size(stim(stimuli(1)),2));  % tones whitout any modulation
mod_tones = nan(size(stimuli,1), size(stimuli,2),  size(stim(stimuli(1)),2));  % tones with rampup/down

%% Main loop
for block = 1:size(raw_tones,1)
    % get main frequency array
    cur_tones               = stim(stimuli(block,:));           % get raw tone     
    raw_tones(block, :, :)  = cur_tones;                        % put into matrix
    mod_tones(block, :, :)  = cur_tones.*w_on_xt.*w_off_xt;     % apply ramp
end

return