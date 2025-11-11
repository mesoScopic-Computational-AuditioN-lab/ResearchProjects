function [raw_tones, mod_tones] = create_tones_tonotopy(stimuli, stim_len, samp_rate, rampup_dur, rampdown_dur, amplitude_mod, mod_index, mod_min, mod_phase)

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
%   - amplitdue_mod :   amplitude modulation carrier freq.           %
%   - mod_index :       modulation index power                       %
%   - mod_min :         minimum of modulation                        %
%   - mod_phase :       modulation phase in degree (or 'random')     %
%                                                                    %
%   Outputs:    returns both raw and modulation tones, that are      %
%               also saved.                                          %
%                                                                    %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Set function handles and precreate arrays

% set function handles for stimulus and modulation
stim    = @(p) [createwaveform(p, stim_len, samp_rate)];
mod     = @(q) [createwaveform(amplitude_mod, stim_len, samp_rate, q)];

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
raw_tones = nan(size(stimuli,2),  size(stim(stimuli(1)),2));  % tones whitout any modulation
mod_tones = nan(size(stimuli,2),  size(stim(stimuli(1)),2));  % tones with modulation and rampup/down

%% Main loop
%(in feature change to single matrix multiplication, see create_tones_main)
for tone = 1:size(raw_tones,1) 
    
    % get main frequency array
    cur_tone            = stim(stimuli(tone));    % get raw tone
    raw_tones(tone, :)  = cur_tone;               % put into matrix

    % apply ramp
    cur_tone            = cur_tone.*w_on_xt.*w_off_xt;
    
    % add modulation
    if strcmp('random',mod_phase)       % check if we want to randomly phaseshift modulation
        mod_phase = rand(1)*360;
    end
    modulator           = ((mod_index * mod(mod_phase)) + 1)/ ...    % create modulatotion waveform
                            (2/(mod_index-mod_min)) + mod_min;    
    cur_tone            = modulator .* cur_tone;                     % apply aplitude modualtion
    
    % save modulated tone
    mod_tones(tone, :) = cur_tone;
end

return