function mainpredsound_pupil(ppnum, wrun, setup, eyetracking)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Fast acces for testing purpos                                                        %
%                                                                                       %
% addpath(genpath([pwd '/function s/']));addpath(genpath([pwd '/stimuli/']));startup1; %
% Screen( 'Preference', 'SkipSyncTests', 1);                                            %
% clear mex
% addpath(genpath([pwd '/functions/'])); startup1;
% ppnum ='1'; wrun ='1'; setup =0;     % set to 2 for fMRI                              %
% opacity = 0.9;  % make window partially see trough                                    %
% PsychDebugWindowConfiguration([], opacity)                                            %
% eyetracking = 0;                                                                      %
%                                                                                       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

settings_main_pupil;

% predefine online trigger tracker variable % add if load
if ~exist([pwd '/data/' ppnum '/' ppnum '-mainpred.mat'],'file')
    online_trigger_tracker = struct('trigger',string(),'t',GetSecs,'timedelay',nan, 'notes',string());
else
    disp('loading in online tracker')
    load(fullfile( pwd, '/data/', ppnum, [ppnum '-mainpred.mat']),'online_trigger_tracker');  
end

%% Eyetracking setup
SETUP_eyetracker(eyetracking);

%% CONDITIONS AND COUNTERBALANCING
nsegments           = cfg.n_segments;                                       % number of segments per block
nblocks             = cfg.n_blocks;                                         % number of blocks
nruns               = cfg.n_runs;                                           % number of runs

nprobs              = cfg.n_probs;                                          % number of probabilities
npairs              = size(cfg.block_pairs, 2);                             % number of freq pairs

nsegments_all       = nsegments * nblocks;                                  % total number of segments      

% for first time loading exp: do counterbalancing, otherwise load
if ~exist(fullfile( pwd, 'stimuli', ppnum, [ppnum '_counterbalancing.mat']),'file')
    
    % get lengths
    len_mainConditions_pre      =   length(cfg.mainConditions_pre_descr);   % length conditions first segment
    len_mainConditions_post     =   length(cfg.mainConditions_post_descr);  % length conditions second segment
    len_distWidths              =   length(cfg.distWidths);                 % length distribution widths
    len_distProbs               =   length(cfg.distProbs);                  % length distribution probability
    len_onsetJitter             =   length(cfg.onsetJitter);                % length jitter of length of first segment
    len_distributions           =   cfg.n_distr;                            % number of distributions of where to sample from
    len_centerfreqs             =   size(cfg.block_pairs,2);                % number of center frequencies
    tpc                         =   cfg.tpc;                                % number trials per conditions (must be divisable by nr secondary condtions)

    % do the counterbalancing
    cb_stim = counterbalance([len_mainConditions_pre, len_mainConditions_post], tpc/(len_onsetJitter*len_distributions), [len_onsetJitter, len_distributions], 1, [len_distWidths, len_distProbs,len_distWidths, len_distProbs, len_centerfreqs], 'subdiv');
    % cb_stim( 1, :) % mainConditions_pre
    % cb_stim( 2, :) % mainConditions_post
    % cb_stim( 3, :) % onsetJitter
    % cb_stim( 4, :) % distribution A or B
    % cb_stim( 5, :) % distWidths_pre
    % cb_stim( 6, :) % distProbs_pre
    % cb_stim( 7, :) % distWidths_post
    % cb_stim( 8, :) % distProbs_post
    % cb_stim( 9, :) % center frequencies
    
    %TEMP HARCODE
    %RAND-REG, REG-RAND, REG-DREG
    %cb_stim( 1, 1:3) = [2, 1, 1];
    %cb_stim( 2, 1:3) = [1, 2, 3];
    
    % save raw counter balancing
    save (fullfile( pwd, 'stimuli', ppnum, [ppnum '_counterbalancing.mat']), 'cb_stim');

else

    % if we are resuming, instead reload counterbalancing
    load (fullfile( pwd, 'stimuli', ppnum, [ppnum '_counterbalancing.mat']), 'cb_stim');
end

% get actual values
distWidths_pre      = cfg.distWidths(cb_stim( 5, :));               % width of first segment
distWidths_post     = cfg.distWidths(cb_stim( 7, :));               % width of second segment
distProbs_pre       = cfg.distProbs(cb_stim( 6, :));                % probability of first segment
distProbs_post      = cfg.distProbs(cb_stim( 8, :));                % probability of second segment
segmentJitter       = cfg.onsetJitter(cb_stim( 3, :));              % jitter of first segemnt in each block

% labbeling setting
sampling_probs = [cfg.REG_sampling_prob, cfg.RAND_sampling_prob nan];       % samplingprob for REG, and RAND, placeholder nan for d' condition
sampling_widths = [cfg.REG_sampling_width, cfg.RAND_sampling_width nan];    % samplingwidth for REG, and RAND, placeholder nan for d' condition
ab_dist = [1 0];                                                            % dominant distribution A vs B

% calculate maincondition prob, and width
sampling_prob_pre = sampling_probs(cb_stim( 1, :));                         % fetch probability pre
mainConditions_pre = abs(sampling_prob_pre - ab_dist(cb_stim( 4, :)));      % get correct A vs B distribution prob
mainConditions_pre_width = sampling_widths(cb_stim( 1, :));                 % fetch sampling width pre

sampling_prob_post = sampling_probs(cb_stim( 2, :));                        % fetch probability post
mainConditions_post = abs(sampling_prob_post - ab_dist(cb_stim( 4, :)));    % get correct A vs B distribution prob
mainConditions_post(isnan(mainConditions_post)) = abs(1 - mainConditions_pre(isnan(mainConditions_post)));                % replace nans (d' condition) with oposite of pre 
mainConditions_post_width = sampling_widths(cb_stim( 2, :));                % fetch sampling width post
mainConditions_post_width(isnan(mainConditions_post_width)) = mainConditions_pre_width(isnan(mainConditions_post_width)); % replace nans (d' condition) with identical of pre 

% apply width changes
mainConditions_wChanges_pre = mainConditions_pre + distProbs_pre;
mainConditions_wChanges_post = mainConditions_post + distProbs_post;
mainConditions_width_wChanges_pre = mainConditions_pre_width + distWidths_pre;
mainConditions_width_wChanges_post = mainConditions_post_width + distWidths_post;

% combine values for later
main_condnr_comb    = NaN(1, size(cb_stim( 1, :),2) + size(cb_stim( 2, :),2));
prob_cond_comb      = NaN(1, size(cb_stim( 6, :),2) + size(cb_stim( 8, :),2));
width_cond_comb     = NaN(1, size(cb_stim( 5, :),2) + size(cb_stim( 7, :),2));
pres_prob_comb      = NaN(1,size(mainConditions_wChanges_pre,2) + size(mainConditions_wChanges_post,2));
pres_width_comb     = NaN(1,size(mainConditions_width_wChanges_pre,2) + size(mainConditions_width_wChanges_post,2));

% populate interleaved - pre, post
main_condnr_comb(1:2:end)   = cb_stim( 1, :);
main_condnr_comb(2:2:end)   = cb_stim( 2, :);
prob_cond_comb(1:2:end)     = cb_stim( 6, :);
prob_cond_comb(2:2:end)     = cb_stim( 8, :);
width_cond_comb(1:2:end)    = cb_stim( 5, :);
width_cond_comb(2:2:end)    = cb_stim( 7, :);
pres_prob_comb(1:2:end)     = mainConditions_wChanges_pre;
pres_prob_comb(2:2:end)     = mainConditions_wChanges_post;
pres_width_comb(1:2:end)    = mainConditions_width_wChanges_pre;
pres_width_comb(2:2:end)    = mainConditions_width_wChanges_post;

% also fetch labels
cfg.cond_labels = cfg.mainConditions_post_descr(main_condnr_comb);

% define order of segments for generation
segmentz = NaN(15, cfg.bpr*nsegments*nruns);
segmentz( 1,  :)     = repelem(1:nruns, nsegments*cfg.bpr);                 % what run nr
segmentz( 2,  :)     = repmat(repelem(1:cfg.bpr, nsegments),1,nruns);       % what block nr
segmentz( 15,  :)    = repelem(1:cfg.bpr * cfg.n_runs, nsegments);          % cumulative block number
segmentz( 3,  :)     = main_condnr_comb;                                    % which trial/segment
segmentz( 4,  :)     = prob_cond_comb;                                      % probability condition used
segmentz( 5,  :)     = width_cond_comb;                                     % width condition used
segmentz( 6,  :)     = pres_prob_comb;                                      % probability of selecting A in section
segmentz( 7,  :)     = 1-pres_prob_comb;                                    % probability of selecting B in section
segmentz( 8,  :)     = pres_width_comb;                                     % sampling width of stimuli (A,B are linked atm)

segmentz( 9,  :)     = repelem(cfg.cent_freqs(cfg.block_pairs(1, cb_stim( 9, :))), cfg.n_segments); % center freq of A this block
segmentz( 10,  :)    = repelem(cfg.cent_freqs(cfg.block_pairs(2, cb_stim( 9, :))), cfg.n_segments); % center freq of B this block

segmentz( 11,  :)    = repelem(segmentJitter, cfg.n_segments);              % onset jitter amount
segmentz( 12,  :)    = repelem(cb_stim( 4, :), cfg.n_segments);             % dominant distribution
segmentz( 13,  :)    = repmat(1:cfg.n_segments, 1, size(segmentz,2)/size(1:cfg.n_segments,2)); % what segment (pre/post) in block
segmentz( 14,  :)    = cfg.tps(segmentz( 13,  :));                          % number of stimuli per segment (raw)
segmentz( 14,  1:2:end) = segmentz( 14,  1:2:end) + segmentJitter;          % append jitter

% define relative timings  
timingz             =   nan( 14, sum(segmentz( 14,  :)));
timingz( 1, :)      =   repelem(segmentz( 1,  :), segmentz( 14,  :));       % [1]: What run
timingz( 2, :)      =   repelem(segmentz( 2,  :), segmentz( 14,  :));       % [2]: What block
timingz( 3, :)      =   cell2mat(arrayfun(@(x) 1:x, segmentz( 14,  :), 'UniformOutput', false));    % [3]: What trial in block
timingz( 4, :)      =   repelem(segmentz( 13,  :), segmentz( 14,  :));      % [4]: What segment (for differently sized segments) 
timingz( 14, :)     =   repelem(segmentz( 15,  :), segmentz( 14,  :));      % [14]: What cumalative block nr
                                                                            % --Here present just for information-- %                                                                           
timingz( 5, :)      =   nan;                                                % [5]: What time should we start present each trial
timingz( 6, :)      =   nan;                                                % [6]: What time should we stop present each trial
timingz( 7, :)      =   nan;                                                % [7]: The real start time of a trial
timingz( 8, :)      =   nan;                                                % [8]: The estimated real stop time of a trial
timingz( 9, :)      =   nan;                                                % [9]: The real presentation length of a trial
timingz( 10, :)     =   nan;                                                % [10]: The enpostion in sec of buffered audio relative to buffered snipped
timingz( 11, :)     =   nan;                                                % [11]: The presented log frequency this trial
timingz( 12, :)     =   nan;                                                % [12]: The presented frequency this trial
timingz( 13, :)     =   nan;                                                % [13]: The endpostion in sec after post segment waiting

%% GENERATE TONES

% generate tones (or load if they were already created)
if ~exist([pwd '/stimuli/' ppnum '/' ppnum '_main_stims.mat'],'file')
    % display massage
    disptext(w, 'No stimuli found, Generating now...', screenrect, ...
                cfg.visual.fontsize, cfg.visual.textfont, cfg.visual.textcol);
    Screen('Flip', w);
    % actually generate tones
    pres_freq               = generate_frequencies_pupil(cfg, segmentz);
    [~, mod_tones]          = create_tones_pupil(pres_freq, cfg.stim_t, cfg.sound.samp_rate, ...
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
    aprox_idx       = linspace(log2(cfg.minfreq), log2(cfg.maxfreq), 500);                   % set frequencies to aproximate (and take closest)
    aprox_loudness  = pchip(log2(equal.freq0), cfg.loudLevel(all_loudness), aprox_idx);      % use piecewise cubic interpolation (instead of spine, to circomvent under/overshoots)
    log_pres_freq   = cellfun(@(x) log2(x), pres_freq, 'UniformOutput', false);                          

    % loop over blocks and adjust tone intencity
    for s = 1:size(mod_tones,2)

        % calculate closest index
        searcharray         = repmat(aprox_idx', [1, length(log_pres_freq{s})]);            % create a repamt search array to use in a one sweep search
        [~, closidx]        = min(abs(searcharray-log_pres_freq{s}));                       % calculate what is the closest value(s index) in interpolated array
        mod_tones{s}        = aprox_loudness(closidx)' .* mod_tones{s};                     % weigh the tones by intensity
    end
else
    sca; ShowCursor;
    error('[!!!] No loudness equalisation file found, please do loudness equalisation first...');
end

% playback length array
itilen                      =   cfg.iti*cfg.sound.samp_rate;                % length of iti
padlen                      =   cfg.padding*cfg.sound.samp_rate;            % length of padding
nplayback                   =   cfg.playbacklength;
halfpad                     =   cfg.padding/2;                              % split padding before and after stim
% what is the iti length
if nplayback > 1       % if we sample multiple together
    playbacklength          =   size(mod_tones{1},2) + padlen + itilen;     % take length + iti
else
    playbacklength          =   size(mod_tones{1},2) + padlen;              % take length without iti for single pres mode
end

% load fixation bull into memory
BullTex(1)     = get_bull_tex_2(w, cfg.visual.bull_eye_col, cfg.visual.bull_in_col, cfg.visual.bull_out_col, cfg.visual.bull_fixrads);
BullTex(2)     = get_bull_tex_2(w, cfg.visual.bull_eye_col, cfg.visual.bull_in_col_cor, cfg.visual.bull_out_col, cfg.visual.bull_fixrads);
BullTex(3)     = get_bull_tex_2(w, cfg.visual.bull_eye_col, cfg.visual.bull_in_col_inc, cfg.visual.bull_out_col, cfg.visual.bull_fixrads);


%% INITIALIZE
try
    PsychPortAudio('Close'); 
catch
    disp('PsychPortAudio already closed, ready for initializing');
end
InitializePsychSound(1);
pahandle = PsychPortAudio('Open', [], 1, 2, cfg.sound.samp_rate, cfg.sound.nrchannels, [], []);

% set keys
nextkey             =   cfg.keys.next;        % key for next trigger
esckey              =   cfg.keys.esckey;      % key for escape key
shiftkey            =   cfg.keys.shiftkey;    % key for shiftkey
space               =   cfg.keys.space;       % key for spacebar

% visualy
bullrect            =   CenterRect([0 0 cfg.visual.bullsize cfg.visual.bullsize], screenrect);

% set relative start timings for later use
presdelay           =   cfg.pressdelay;                                     % schedule some time in advance to garantee timings
blockWait           =   repmat([cfg.wait_onset  repmat(cfg.ibi, 1, cfg.bpr - 1)], 1, cfg.n_runs); % time to wait for each block

% save pres frequency info (in hz) into timingz matrix
timingz( 11, : )     =   cell2mat(log_pres_freq);                           % [11]: The presented log frequency this trial
timingz( 12, : )     =   cell2mat(pres_freq);                               % [12]: The presented frequency this trial

%% RUN TRIALS
rblk                =   (str2double(wrun)-1)*cfg.bpr + 1;                   % set at what block to start
responses           =   [];                                                 % empty, for future expension

% load saved timingz of previous runs
if rblk > 1
    load (fullfile( pwd, 'data', ppnum, [ppnum '-mainpred.mat']), 'responses', 'segmentz', 'timingz');
end

% present welcome message untill keypress
multilinetext(w, cfg.visual.waitforstartexp, screenrect, ...
              equal.fontsize, equal.textfont, equal.textcol, 1.2, [3]);
Screen('Flip', w);
waitforresponse(nextkey, esckey, shiftkey, segmentz, responses, timingz, online_trigger_tracker);  waitfornokey;

% loop over runs (with possible resume point)
for crun = str2double(wrun):cfg.n_runs
    
    % sellect block range for run
    rblk                =   (crun-1)*cfg.bpr + 1;                                  % adjust block
    
    % give run information
    multilinetext(w, {['Block ' num2str(crun) '/' num2str(cfg.n_runs)], ' ', cfg.visual.waitforstartblock{1}}, screenrect, ...
                  equal.fontsize, equal.textfont, equal.textcol, 1.2, []);
    Screen('Flip', w);
    waitforresponse(nextkey, esckey, shiftkey, segmentz, responses, timingz, online_trigger_tracker);  waitfornokey;
    
    % check if eyetracking 
    fixateBool = false;
    while ~fixateBool
        fixateBool = checkFixation(BullTex(1), bullrect, 4,false,eyetracking);
        if fixateBool == true
            disp('Fixation checking successfull.')
        else
            multilinetext(w, {['Block ' num2str(crun) '/' num2str(cfg.n_runs)], ' ', 'Pupil data not readable, please re-calibrate!'}, screenrect, ...
                  equal.fontsize, equal.textfont, equal.textcol, 1.2, []);
            Screen('Flip', w);
            waitforresponse(nextkey, esckey, shiftkey);  waitfornokey;

        end
    end

    
    % send trigger of start of run
    [online_trigger_tracker.t(end+1), online_trigger_tracker.trigger(end+1), online_trigger_tracker.timedelay(end+1)] = vpx_custom_trigger('R', eyetracking);
    online_trigger_tracker.notes(end+1) = join(['R',num2str(crun), ' start']);
    

    % run over blocks for this run
    for block = rblk:rblk+cfg.bpr-1 
        %% Prepair things in TRs before a block starts
    
        % calculate length of current block
        len_cblock      =   sum(segmentz(14,  segmentz(15,:) == block));

        % set new temporary arrays for this block
        start_times     =   nan(1, len_cblock);
        est_stops       =   nan(1, len_cblock);
        real_pres_len   =   nan(1, len_cblock);
        all_endpos      =   nan(1, len_cblock);

        % calculate relative onset timeings
        reltime             =   0:len_cblock-1;                                     % set relative time array
        reltime             =   reltime*(cfg.stim_t+cfg.iti);                       % at what relative time to actually present each stimuli
        reltime             =   reltime + presdelay - halfpad;                      % alter relative time to add presentation delay (and to ignore padding)
        expendpos           =   linspace(cfg.stim_t + cfg.padding, (cfg.stim_t + cfg.padding) * len_cblock, len_cblock); % calculate expected endpositions in order to resync over/undershoots

        % start waiting for triggers
        waitforsc      = blockWait(block);     % get waiting period for this block
        while waitforsc > 0
    
            % start measuring pulses
            blockstarttime      = WaitSecs(1); 
    
            % draw fixation
            Screen('DrawTexture',w,BullTex(1),[],bullrect);
            [~] = Screen('Flip', w);
    
            % prepair things at the right time
            if waitforsc <= 1 && waitforsc > 0                     % when the last pulse before stim onset occured
    
                % prepair new block
                timingz( 5, timingz(14, :) == block)        =   reltime + blockstarttime;                           % [5]: What time should we start present each trial
                timingz( 6, timingz(14, :) == block)        =   reltime + blockstarttime + cfg.stim_t + cfg.padding;% [6]: What time should we stop present each trial
                start_sched                                 =   timingz( 5, timingz(14, :) == block);               % for faster loading save it also in a small temp array
                stop_sched                                  =   timingz( 6, timingz(14, :) == block);               % idem
                
            elseif waitforsc == blockWait(block)           % after the first waiting pulse place everything in buffer
    
                % place current block into the audio buffer
                blckidx                                     =   find(segmentz(15,:) == block);                      % define end and start index of block
                p                                           =   padlen/2;
                blockAudio                                  =   zeros(len_cblock, playbacklength);                  % predefine zero array
                blockAudio(:, p+1:size(mod_tones{1},2)+p)   =   vertcat(mod_tones{blckidx(1):blckidx(end)});        % take waveform data from this block
                if nplayback > 1
                        blockAudio          =   reshape(blockAudio', [], size(blockAudio,1)/nplayback)';            % if nplayback is longer then single trial, add iti in buffer
                        blockAudio          =   blockAudio(:,1:end-itilen);                                         % and remove silent periode between snippets
                end
                blockAudio                                  =   squeeze(reshape(blockAudio', 1, []));               % take long format

                PsychPortAudio('FillBuffer', pahandle, blockAudio);                                                 % fill buffer with this array
            end
    
            % count
            waitforsc                                       =   waitforsc -1;          % countdown waitTRs
        end
        
%         block_trigger = join(['B' ,num2str(block)]); % cfg.trigger.blockOnset+
        [online_trigger_tracker.t(end+1), online_trigger_tracker.trigger(end+1), online_trigger_tracker.timedelay(end+1)] = vpx_custom_trigger(join(['B' ,num2str(block)]), eyetracking);
        online_trigger_tracker.notes(end+1) = join(['B',num2str(block), ' start']);


        %% Main presentation phase of block
        % loop over trials in block
        sync_del            =   0;                      % compensate for over/undershoots of bufferplay
        for trial           =   1:nplayback:len_cblock  % go throught trials in steps of nplaybacklength
            trialend        =   trial+nplayback-1;      % get where we end (if nplaybacklength is 1, this is identical to 'trial')
            prepostorder    =   [0 diff(timingz( 4, timingz(14, :) == block))];

            % define what trigger to send
            if trial == 1                                                                           % if start of block
                sendtrg = cfg.trigger.firstSegOnset;
            elseif prepostorder(trial)
                sendtrg = cfg.trigger.secSegOnset;
            else
                sendtrg = cfg.trigger.stimOnset;
            end

            % use scheduled start and stop times to start/stop playing 
            PsychPortAudio('Start', pahandle, 1, ...                                                % start/stop audio playback at predefined timings
                            start_sched(trial), ...
                            1, ...
                            stop_sched(trialend) + sync_del, 1);   
            % send trigger
            [online_trigger_tracker.t(end+1), online_trigger_tracker.trigger(end+1), online_trigger_tracker.timedelay(end+1)] = vpx_custom_trigger('S', eyetracking);
            online_trigger_tracker.notes(end+1) = join(['S',num2str(trial)]);
            
            % stop at correct time
            [startTime, endPos, ~, estStop]    = PsychPortAudio('Stop', pahandle, 3);               % stop when finished and save timings
            
            % save trial timings
            start_times(trial)          = startTime;            % real start times of trial
            est_stops(trialend)         = estStop;              % estimated real stop times of trial
            real_pres_len(trial)        = estStop-startTime;    % real presentation length of trial
            all_endpos(trialend)        = endPos;               % endpostion in sec of buffered audio relative to buffer
    
            % compensate for current delay
            sync_del                    = expendpos(trial) - endPos;
        end
    
        %% PUT IN FUNCTION
        % send trigger
        [online_trigger_tracker.t(end+1), online_trigger_tracker.trigger(end+1), online_trigger_tracker.timedelay(end+1)] = vpx_custom_trigger('E', eyetracking);
        online_trigger_tracker.notes(end+1) = join(['B',num2str(block), ' end']);
        
        %% save timings of this block into the full array
        timingz( 7, timingz(14, :) == block)            =   start_times;          % [7]: The real start time of a trial 
        timingz( 8, timingz(14, :) == block)            =   est_stops;            % [8]: The estimated real stop time of a trial
        timingz( 9, timingz(14, :) == block)            =   real_pres_len;        % [9]: The real presentation length of a trial
        timingz( 10, timingz(14, :) == block)           =   all_endpos;           % [10]: The enpostion in sec of buffered audio relative to buffered snipped
        
    end

    % wait after ofset of last block in run
    WaitSecs(cfg.ibi); 

    %% END OF BLOCK
    save (fullfile( pwd, 'data', ppnum, [ppnum '-mainpred.mat']), 'responses', 'segmentz', 'timingz', 'cfg', 'online_trigger_tracker');         % already save data (for possible crashes etc.)

    % present end of block text
    multilinetext(w, cfg.visual.endofblock, screenrect, ...
                  equal.fontsize, equal.textfont, equal.textcol, 1.2, []);
    Screen('Flip', w);
    waitforresponse(nextkey, esckey, shiftkey, segmentz, responses, timingz, online_trigger_tracker);  waitfornokey;

end

%% close eyetracker datafile
% send closing trigger
[online_trigger_tracker.t(end+1), online_trigger_tracker.trigger(end+1), online_trigger_tracker.timedelay(end+1)] = vpx_custom_trigger('F', eyetracking, true);
online_trigger_tracker.notes(end+1) = 'Experiment end';

%% SHUT DOWN AND SAVE DATA
save (fullfile( pwd, 'data', ppnum, [ppnum '_mainpred.mat']), 'responses', 'segmentz', 'timingz', 'cfg', 'online_trigger_tracker');
PsychPortAudio('Close'); 

end