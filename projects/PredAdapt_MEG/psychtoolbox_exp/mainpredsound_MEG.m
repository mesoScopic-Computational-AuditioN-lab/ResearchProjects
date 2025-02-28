% function mainpredsound_MEG(ppnum, wrun, setup)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Fast acces for testing purpos                                                        %
%                                                                                       %
addpath(genpath([pwd '/function s/']));addpath(genpath([pwd '/stimuli/']));%startup1;   %
Screen( 'Preference', 'SkipSyncTests', 1);                                            %
ppnum ='999'; wrun ='1'; setup =0;     % set to 2 for fMRI                            %
%                                                                                       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%sca

settings_main;

%% CONDITIONS AND COUNTERBALANCING
addpath(genpath([pwd '/functions/']));      % add functions to path

nsegments           = cfg.n_segments;                                       % number of segments per block
nblocks             = cfg.n_blocks;                                         % number of blocks
nruns               = cfg.MEG.n_runs;                                           % number of runs

nadpts              = 3;                                                    % number of adaptation cohorts
npreds              = 3;                                                    % number of prediction cohorts

nprobs              = cfg.n_probs;                                          % number of probabilities
npairs              = size(cfg.block_pairs, 2);                             % number of freq pairs

nsegments_all       = nsegments * nblocks;                                  % total number of segments      

% for first time loading exp: do counterbalancing, otherwise load
if str2double(wrun) == 1

% block counterbalance
cb_blocks           = counterbalance([npairs], nblocks/npairs, [], 1, [], 'subdiv');
cb_segments         = counterbalance([nprobs], 1, [], nsegments_all/nprobs, [], 'full');

segmentz = NaN(11, nblocks*nsegments);
segmentz( 1,  :)     = repelem(1:nruns, nsegments*cfg.MEG.bpr);                 % In MEG we can see this as bigger blocks
segmentz( 2,  :)     = repelem(1:nblocks, nsegments);                       % what block
segmentz( 3,  :)     = repmat(1:nsegments, 1, nblocks);                     % trial in block
segmentz( 4,  :)     = cb_segments;                                         % prob condition per segment
segmentz( 5,  :)     = repelem(cb_blocks, nsegments);                       % what freq pair is being presented

segmentz( 6,  :)     = cfg.stim_probs(1, segmentz(4, :));                   % probability of selecting A in section
segmentz( 7,  :)     = cfg.stim_probs(2, segmentz(4, :));                   % probability of selecting B in section

segmentz( 8,  :)     = cfg.cent_freqs(cfg.block_pairs(1, segmentz(5,:)));   % center freq of A this block
segmentz( 9,  :)     = cfg.cent_freqs(cfg.block_pairs(2, segmentz(5,:)));   % center freq of B this block

% aproximate predictablity and adaptability of blocks
pred_idx = nan(1,nblocks);
adpt_idx = nan(1,nblocks);
for b = 1:nblocks
    pred_idx(b) = std(diff(cfg.stim_probs(1, segmentz(4, segmentz(2,:) == b))));
    adpt_idx(b) = mean(cfg.stim_probs(1, segmentz(4, segmentz(2,:) == b)));
end
[~, pred_idx] = sort(pred_idx);
[~, adpt_idx] = sort(adpt_idx);

% set segments predictability
for i = 1:nadpts
    segmentz(10, ismember(segmentz(2, :), pred_idx(repelem(1:nadpts, nblocks/nadpts) == i)))     = i;    % pedictability (1=low-switching, 2=medium-swithing, 3=high-switching)
    segmentz(11, ismember(segmentz(2, :), adpt_idx(repelem(1:npreds, nblocks/npreds) == i)))     = i;    % adaptability (1=A-higher, 2=average, 3=B-higher)
end

% define relative timings  
timingz             =   nan( 12, nblocks * cfg.tpb);
timingz( 1, :)      =   repelem(1:nruns, cfg.tpb*cfg.MEG.bpr);                  % [1]: What run
timingz( 2, :)      =   repelem(1:nblocks, cfg.tpb);                        % [2]: What block
timingz( 3, :)      =   repmat(1:cfg.tpb, 1, nblocks);                      % [3]: What trial in block
timingz( 4, :)      =   repmat(repelem(1:nsegments, cfg.tps), 1, nblocks);  % [4]: What segment
                                                                            % --Here present just for information-- %
timingz( 5, :)      =   nan;                                                % [5]: What time should we start present each trial
timingz( 6, :)      =   nan;                                                % [6]: What time should we stop present each trial
timingz( 7, :)      =   nan;                                                % [7]: The real start time of a trial
timingz( 8, :)      =   nan;                                                % [8]: The estimated real stop time of a trial
timingz( 9, :)      =   nan;                                                % [9]: The real presentation length of a trial
timingz( 10, :)     =   nan;                                                % [10]: The enpostion in sec of buffered audio relative to buffered snipped
timingz( 11, :)     =   nan;                                                % [11]: The presented log frequency this trial
timingz( 12, :)     =   nan;                                                % [12]: The presented frequency this trial

end

%% GENERATE TONES

% generate tones (or load if they were already created)
if ~exist([pwd '/stimuli/' ppnum '/' ppnum '_main_stims.mat'],'file')
    % display massage
    disptext(w, 'No stimuli found, Generating now...', screenrect, ...
                cfg.visual.fontsize, cfg.visual.textfont, cfg.visual.textcol);
    Screen('Flip', w);
    % actually generate tones
    pres_freq               = generate_frequencies_main(cfg, segmentz);
    [~, mod_tones]          = create_tones_main(pres_freq, cfg.stim_t, cfg.sound.MEG.samp_rate, ...
                                                    cfg.ramp_ops, cfg.ramp_dns);
    save (fullfile( pwd, 'stimuli', ppnum, [ppnum '_main_stims.mat']), 'mod_tones', 'pres_freq', 'segmentz');
else
    % display massage
    disptext(w, 'Loading stimuli', screenrect, ...
                cfg.visual.fontsize, cfg.visual.textfont, cfg.visual.textcol);
    Screen('Flip', w);
    % actually load tones from disk
    load (fullfile( pwd, 'stimuli', ppnum, [ppnum '_main_stims.mat']), 'mod_tones', 'pres_freq', 'segmentz');
end

% apply equalised loudness curves
if exist([pwd '/loudness/' ppnum '/' ppnum '-loudness.mat'],'file')

    % load loudness values for participant
    load(fullfile( pwd, 'loudness', ppnum, [ppnum '-loudness.mat']),'all_loudness', 'equal');  

    % loudnesses
    aprox_idx       = linspace(log2(cfg.minfreq), log2(cfg.maxfreq), cfg.tpb);                   % set frequencies to aproximate (and take closest)
    aprox_loudness  = pchip(log2(equal.freq0), cfg.loudLevel(all_loudness), aprox_idx);          % use piecewise cubic interpolation (instead of spine, to circomvent under/overshoots)
    log_pres_freq   = log2(pres_freq);                          

    % loop over blocks and adjust tone intencity
    for blk = 1: size(log_pres_freq,1)

        % calculate closest index
        searcharray         = repmat(aprox_idx', [1, length(log_pres_freq(blk,:))]);            % create a repamt search array to use in a one sweep search
        [~, closidx]        = min(abs(searcharray-log_pres_freq(blk,:)));                       % calculate what is the closest value(s index) in interpolated array
        mod_tones(blk,:,:)  = aprox_loudness(closidx)' .* squeeze(mod_tones(blk,:,:));          % weigh the tones by intensity
    end
else
    sca; ShowCursor;
    error('[!!!] No loudness equalisation file found, please do loudness equalisation first...');
end

% load left right equalisation load
if exist([pwd '/loudness/' ppnum '/' ppnum '-rl_loudness.mat'],'file')
    load(fullfile( pwd, 'loudness', ppnum, [ppnum '-rl_loudness.mat']),'relloudnes');           % load right left loudness values for participant
else
    relloudnes = [1, 1];                                                                        % set right left loudness to ones
    warning('No right/left equalisation done, equal loudness assumed - [1, 1]');
end

% playback length array
itilen                      =   cfg.iti*cfg.sound.MEG.samp_rate;                % length of iti
padlen                      =   cfg.padding*cfg.sound.MEG.samp_rate;            % length of padding
nplayback                   =   cfg.playbacklength;
halfpad                     =   cfg.padding/2;                              % split padding before and after stim
% what is the iti length
if nplayback > 1       % if we sample multiple together
    playbacklength          =   size(mod_tones,3) + padlen + itilen;        % take length + iti
else
    playbacklength          =   size(mod_tones,3) + padlen;                 % take length without iti for single pres mode
end

% load fixation bull into memory
BullTex(1)     = get_bull_tex_2(w, cfg.visual.bull_eye_col, cfg.visual.bull_in_col, cfg.visual.bull_out_col, cfg.visual.bull_fixrads);
BullTex(2)     = get_bull_tex_2(w, cfg.visual.bull_eye_col, cfg.visual.bull_in_col_cor, cfg.visual.bull_out_col, cfg.visual.bull_fixrads);
BullTex(3)     = get_bull_tex_2(w, cfg.visual.bull_eye_col, cfg.visual.bull_in_col_inc, cfg.visual.bull_out_col, cfg.visual.bull_fixrads);


%% INITIALIZE
PsychPortAudio('Close'); 
InitializePsychSound(1);
pahandle = PsychPortAudio('Open', [], 1, 2, cfg.sound.MEG.samp_rate, 1, [], []);
B = cfg.setup.B;

% set keys
nextkey             =   cfg.keys.next;        % key for next trigger
esckey              =   cfg.keys.esckey;      % key for escape key
shiftkey            =   cfg.keys.shiftkey;    % key for shiftkey

% visualy
bullrect            =   CenterRect([0 0 cfg.visual.bullsize cfg.visual.bullsize], screenrect);

% set relative start timings for later use
reltime             =   0:cfg.tpb-1;                                        % set relative time array
reltime             =   reltime*(cfg.stim_t+cfg.iti);                       % at what relative time to actually present each stimuli
presdelay           =   cfg.MEG.pressdelay;                                 % schedule one TR in advance to garantee timings
reltime             =   reltime + presdelay - halfpad;                      % alter relative time to add presentation delay (and to ignore padding)
blockWait           =   repmat([cfg.MEG.wait_onset  repmat(cfg.MEG.ibi, 1, cfg.MEG.bpr - 1)], 1, cfg.MEG.n_runs); % time to wait for each block
expendpos           =   linspace(cfg.stim_t + cfg.padding, (cfg.stim_t + cfg.padding) * cfg.tpb, cfg.tpb); % calculate expected endpositions in order to resync over/undershoots

% save pres frequency info (in hz) into timingz matrix
timingz( 11, :)     =   reshape(log_pres_freq.',1,[]);                      % [11]: The presented log frequency this trial
timingz( 12, :)     =   reshape(pres_freq.',1,[]);                          % [12]: The presented frequency this trial

%% RUN CLICK SOUND FOR FAILSAFE SYNCRONISATION

% generate the sound
duration = 0.01; % Duration of the click (seconds) - 10ms
click = [ones(1, round(cfg.sound.MEG.samp_rate*duration)), zeros(1, round(cfg.sound.MEG.samp_rate*duration))]; % Generate the click

% put sound in buffer
PsychPortAudio('FillBuffer', pahandle, click); 

% play the sound 3 times with 0.5 seconds intervals
for i = 1:3
    triggerTime = GetSecs();
    [clickonset]  = PsychPortAudio('Start', pahandle, [], triggerTime+0.1); % play audio at timing (waitForStart>1?)
    cfg.setup.B.sendTrigger(i+3);
    WaitSecs(0.5); % pause for 0.5 seconds before playing again
end

%% RUN TRIALS
rblk                =   (str2double(wrun)-1)*cfg.MEG.bpr + 1;                   % set at what block to start
responses           =   [];                                                 % empty, for future expension

% load saved timingz of previous runs
if rblk > 1
    load (fullfile( pwd, 'data', ppnum, [ppnum '-mainpred.mat']), 'responses', 'segmentz', 'timingz');
end

% present waiting for scanner message till first pulse
multilinetext(w, cfg.visual.waitforstartexp, screenrect, ...
              equal.fontsize, equal.textfont, equal.textcol, 1.2, [3]);
Screen('Flip', w);
waitforbitsi(nextkey, esckey, shiftkey, B, segmentz, responses, timingz);  waitfornokey;

% loop over runs (with possible resume point)
for crun = str2double(wrun):cfg.MEG.n_runs

    % sellect block range for run
    rblk                =   (crun-1)*cfg.MEG.bpr + 1;                                                          % adjust block

    % give run information
    multilinetext(w, {['Block ' num2str(crun) '/' num2str(cfg.MEG.n_runs)], ' ', cfg.visual.waitforstartblock{1}}, screenrect, ...
                  equal.fontsize, equal.textfont, equal.textcol, 1.2, []);
    Screen('Flip', w);
    waitforbitsi(nextkey, esckey, shiftkey, B, segmentz, responses, timingz);  waitfornokey;

    % run over blocks for this run
    for block = rblk:rblk+cfg.MEG.bpr-1 

        %% Prepair things in TRs before a block starts
    
        % set new temporary arrays for this block
        start_times     =   nan(1, cfg.tpb);
        est_stops       =   nan(1, cfg.tpb);
        real_pres_len   =   nan(1, cfg.tpb);
        all_endpos      =   nan(1, cfg.tpb);
    
        % start waiting for triggers
        waitforsc      = blockWait(block);     % get waiting period for this block
        while waitforsc > 0
    
            % start measuring pulses
            blockstarttime      = WaitSecs(1); 
    
            % draw fixation
            Screen('DrawTexture',w,BullTex(1),[],bullrect);
            [~] = Screen('Flip', w);
    
            % prepair things at the right time
            if waitforsc == 1                              % when the last pulse before stim onset occured
    
                % prepair new block
                timingz( 5, timingz(2,:) == block)          =   reltime + blockstarttime;                           % [5]: What time should we start present each trial
                timingz( 6, timingz(2,:) == block)          =   reltime + blockstarttime + cfg.stim_t + cfg.padding;% [6]: What time should we stop present each trial
                start_sched                                 =   timingz( 5, timingz(2,:) == block);                 % for faster loading save it also in a small temp array
                stop_sched                                  =   timingz( 6, timingz(2,:) == block);                 % idem
                
            elseif waitforsc == blockWait(block)           % after the first waiting pulse place everything in buffer
    
                % place current block into the audio buffer
                p                                           =   padlen/2;
                blockAudio                                  =   zeros(size(mod_tones,2), playbacklength);           % predefine zero array
                blockAudio(:, p+1:size(mod_tones,3)+p)      =   squeeze(mod_tones(block,:,:));                      % take waveform data from this block
                if nplayback > 1
                        blockAudio          =   reshape(blockAudio', [], size(blockAudio,1)/nplayback)';            % if nplayback is longer then single trial, add iti in buffer
                        blockAudio          =   blockAudio(:,1:end-itilen);                                         % and remove silent periode between snippets
                end
                blockAudio                                  =   squeeze(reshape(blockAudio', 1, []));               % take long format
                blockAudio                                  =   relloudnes' .* blockAudio;                          % modify intensity left / right

                PsychPortAudio('FillBuffer', pahandle, blockAudio);                                                 % fill buffer with this array
            end
    
            % count
            waitforsc                                       =   waitforsc -1;          % countdown waitTRs
        end
    
        %% Main presentation phase of block
        % loop over trials in block
        sync_del            =   0;                      % compensate for over/undershoots of bufferplay
        for trial           =   1:nplayback:cfg.tpb     % go throught trials in steps of nplaybacklength
            trialend        =   trial+nplayback-1;      % get where we end (if nplaybacklength is 1, this is identical to 'trial')

            % define what trigger to send
            if trial == 1                                                                           % if start of block
                sendtrg = cfg.MEG.trigger.blockOnset + block;
            elseif rem(trial-1, cfg.tps) == 0                                                       % if start of segment
                sendtrg = cfg.MEG.trigger.segmentOnset + timingz( 4, (timingz(2,:) == block) & (timingz(3,:) == trial));
            else
                sendtrg = cfg.MEG.trigger.stimOnset;
            end

            % use scheduled start and stop times to start/stop playing
            PsychPortAudio('Start', pahandle, 1, ...                                                % start/stop audio playback at predefined timings
                            start_sched(trial), ...
                            1, ...
                            stop_sched(trialend) + sync_del, 1);  
            cfg.setup.B.sendTrigger(sendtrg);                                                       % sent on-trigger to MEG system  
            % stop at correct time
            [startTime, endPos, ~, estStop]    = PsychPortAudio('Stop', pahandle, 3);               % stop when finished and save timings
            cfg.setup.B.sendTrigger(cfg.MEG.trigger.stimOffset);                                    % set off-trigger to MEG system

            % save trial timings
            start_times(trial)          = startTime;            % real start times of trial
            est_stops(trialend)         = estStop;              % estimated real stop times of trial
            real_pres_len(trial)        = estStop-startTime;    % real presentation length of trial
            all_endpos(trialend)        = endPos;               % endpostion in sec of buffered audio relative to buffer
    
            % compensate for current delay
            sync_del                    = expendpos(trial) - endPos;
        end
    
        % get end of block indication
        cfg.setup.B.sendTrigger(cfg.MEG.trigger.blockOffset)    % end of block trigger

        %% save timings of this block into the full array
        timingz( 7, timingz(2,:) == block)            =   start_times;          % [7]: The real start time of a trial 
        timingz( 8, timingz(2,:) == block)            =   est_stops;            % [8]: The estimated real stop time of a trial
        timingz( 9, timingz(2,:) == block)            =   real_pres_len;        % [9]: The real presentation length of a trial
        timingz( 10, timingz(2,:) == block)           =   all_endpos;           % [10]: The enpostion in sec of buffered audio relative to buffered snipped
    
    end

    %% END OF BLOCK
    save (fullfile( pwd, 'data', ppnum, [ppnum '-mainpred.mat']), 'responses', 'segmentz', 'timingz', 'cfg', 'B');         % already save data (for possible crashes etc.)

    % present end of block text
    multilinetext(w, cfg.visual.endofblock, screenrect, ...
                  equal.fontsize, equal.textfont, equal.textcol, 1.2, []);
    Screen('Flip', w);
    waitforbitsi(nextkey, esckey, shiftkey, B, segmentz, responses, timingz);  waitfornokey;

end


%% SHUT DOWN AND SAVE DATA
save (fullfile( pwd, 'data', ppnum, [ppnum '-mainpred.mat']), 'responses', 'segmentz', 'timingz', 'cfg', 'B');
PsychPortAudio('Close'); 
cfg.setup.B.close();

% end