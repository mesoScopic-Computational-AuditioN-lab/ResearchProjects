function mainlocalizer(ppnum, wrun, setup)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Fast acces for testing purpos                                                        %
%                                                                                       %
% addpath(genpath([pwd '/functions/']));addpath(genpath([pwd '/stimuli/']));startup1;   %
% Screen( 'Preference', 'SkipSyncTests', 1);                                            %
% ppnum ='997'; wrun ='1'; setup =0; % set to 2 for fMRI                                  %
%                                                                                       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% CREATE STIMULI
settings_localizer;

% load condition matrix
conditions              =   nan(6, cfg.tpb*cfg.n_blocks);
conditions( 1, :)       =   repelem(linspace(1, cfg.n_blocks, cfg.n_blocks), 1, cfg.tpb);             % [BLOCK] : set which block we are
conditions( 2, :)       =   cfg.stims;                                                                % [FREQ] : what frequency at what trial
conditions( 3, :)       =   cfg.stims == 0;                                                           % [ISGAP] : is sillent gap
conditions( 4, :)       =   cfg.stim_order;                                                           % [SEQ_ORD] : what sequential order was presented
conditions( 5, :)       =   repmat(linspace(1, cfg.tpb, cfg.tpb), 1, cfg.n_blocks);                   % [TRIALBLOCK] : define trial within blokc
conditions( 6, :)       =   cfg.wait_onset + ...                                                      % [EXPTIME] : expected relative timing for stimuli onset
                                cumsum(cfg.jitters) + ...
                                linspace(0, (cfg.stim_len * length(cfg.jitters)) - cfg.stim_len, length(cfg.jitters)) + ...
                                ((conditions(1, :)-1) * cfg.ibi);

% generate tones (or load if they were already created)
if ~exist([pwd '/stimuli/' ppnum '/' ppnum '-r' wrun '_localizer_stims.mat'],'file')
    [raw_tones, mod_tones] = create_tones_tonotopy(2.^cfg.freq_array, cfg.stim_len, cfg.sound.samp_rate, ...        % generate tones
                                                    cfg.rampup_dur, cfg.rampdown_dur, 0, ...
                                                    1, 1, 0);
    save (fullfile( pwd, 'stimuli', ppnum, [ppnum '-r' wrun '_localizer_stims.mat']), 'raw_tones', 'mod_tones');
else
    load (fullfile( pwd, 'stimuli', ppnum, [ppnum '-r' wrun '_localizer_stims.mat']), 'raw_tones', 'mod_tones');
end

% apply equalised loudness curves
if exist([pwd '/loudness/' ppnum '/' ppnum '-loudness.mat'],'file')
    load(fullfile( pwd, 'loudness', ppnum, [ppnum '-loudness.mat']),'all_loudness', 'equal');   
    aprox_loudness = pchip(log2(equal.freq0), cfg.loudLevel(all_loudness), cfg.freq_array);     % use piecewise cubic interpolation (instead of spine, to circomvent under/overshoots)
    mod_tones      = aprox_loudness' .* mod_tones;                                              % apply loudness adjustment in one go
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


% calculate amount of trs and set triggers
cfg.n_s         = size(conditions(1,:), 2) + cfg.n_blocks-1 + 2;
cfg.trg         = zeros(1, cfg.n_s);
cfg.waittimes   = zeros(1, cfg.n_s);
for b = 1:cfg.n_blocks
    sp = 1 + (b - 1) + sum(conditions(1, :) < b) + 1;      % calculate startpoint for triggers
    ep = 1 + (b - 1) + sum(conditions(1, :) <= b);         % calculate endpoint for triggers
    cfg.trg(sp:ep) = 1;                                    % save on what triggers we present anything 
end
cfg.trg(cfg.trg == 1)       = cumsum(cfg.trg(cfg.trg == 1));    % define trgs to which trial
cfg.buffer                  = circshift(cfg.trg, -1);           % save when we are loading things into the buffer

% set waittimes
cfg.waittimes(cfg.trg > 0)  = cfg.jitters;                      % fill in variable jittered isi
cfg.waittimes(1)            = cfg.wait_onset;                   % add onset waiting
cfg.waittimes(end)          = cfg.wait_offset;                  % add offset waiting
cfg.waittimes(cfg.waittimes == 0)  =  cfg.ibi;                  % fill inter blocks with ibi

% load fixation bull into memory
BullTex(1)     = get_bull_tex_2(w, cfg.visual.bull_eye_col, cfg.visual.bull_in_col, cfg.visual.bull_out_col, cfg.visual.bull_fixrads);
BullTex(2)     = get_bull_tex_2(w, cfg.visual.bull_eye_col, cfg.visual.bull_in_col_cor, cfg.visual.bull_out_col, cfg.visual.bull_fixrads);
BullTex(3)     = get_bull_tex_2(w, cfg.visual.bull_eye_col, cfg.visual.bull_in_col_inc, cfg.visual.bull_out_col, cfg.visual.bull_fixrads);

%% INITIALIZE
PsychPortAudio('Close'); 
InitializePsychSound(1);
pahandle = PsychPortAudio('Open', [], 1, [], cfg.sound.samp_rate, cfg.sound.nrchannels, [], cfg.sound.max_latancy);
B = cfg.setup.B;

% set keys
nextkey     = cfg.keys.next;      % key for next trigger
esckey      = cfg.keys.esckey;      % key for escape key
shiftkey    = cfg.keys.shiftkey;    % key for shiftkey

% visualy
bullrect    = CenterRect([0 0 cfg.visual.bullsize cfg.visual.bullsize], screenrect);


%% RUN TRIALS
waittimes     = cfg.waittimes;
ntrials       = size(conditions, 2);
timingz       = NaN(6, ntrials);
responses     = [];   % empty, for future expension

% waiting for first pulse from scanner to start
multilinetext(w, cfg.visual.waitforstartexp, screenrect, ...
              equal.fontsize, equal.textfont, equal.textcol, 1.2, [3]);
Screen('Flip', w);
waitforbitsi(nextkey, esckey, shiftkey, B, responses, timingz);  waitfornokey;
cfg.setup.B.sendTrigger(cfg.MEG.trigger.blockOnset + str2double(wrun));              % sent on-trigger to MEG system 
%% HIER IPV BLOCKONSET EEN NIEUW NUMMER VOOR LOCALIZE ZODAT DEZE MAKKELIJK GESPLITS KUNNEN WORDEN

% loop over triggers
for pulse = 1:size(cfg.trg, 2)
    %% WAIT FOR TRIGGER AND MAIN PRESENTATION

    % wait for trigger
    triggerTime = WaitSecs(waittimes(pulse));

    % play audio from buffer when condition
    if cfg.trg(pulse) > 0  &&  conditions(4, cfg.trg(pulse)) > 0
        cfg.setup.B.sendTrigger(cfg.MEG.trigger.stimOnset);                                     % set off-trigger to MEG system
        [eststatTime]  = PsychPortAudio('Start', pahandle, [], triggerTime+cfg.presdelay);      % play audio at timing (waitForStart>1?)
        cfg.setup.B.sendTrigger(cfg.MEG.trigger.stimOffset);                                    % set off-trigger to MEG system
    end

    %% REFILL BUFFER WHEN NEEDED

    % check if next trigger is a condition in order to prefill buffer
    if cfg.buffer(pulse) > 0  &&  conditions(4, cfg.buffer(pulse)) > 0
        PsychPortAudio('FillBuffer', pahandle, relloudnes' .* mod_tones(conditions(4, cfg.buffer(pulse)), :));     % (re)fill buffer
    end


    %% END OF PRESENTATION GET TIMINGS

    % stop audio playing and save timings
    [startTime, ~, ~, estStopTime]    = PsychPortAudio('Stop', pahandle, 1);     % save audio presentation timings

    % draw fixation
    Screen('DrawTexture',w,BullTex(1),[],bullrect);
    [~] = Screen('Flip', w);

    %% REGISTER TIMINGS

    % register tone presentation per stimulus
    if cfg.trg(pulse) > 0  &&  conditions(4, cfg.trg(pulse)) > 0
        timingz( 1, cfg.trg(pulse))       =    triggerTime;                  % pulse onset timing
        timingz( 2, cfg.trg(pulse))       =    startTime;                    % estimated audio onset timing
        timingz( 3, cfg.trg(pulse))       =    estStopTime;                  % estimated audio offset timing
        timingz( 4, cfg.trg(pulse))       =    estStopTime - startTime;      % estimated presentation timing
        timingz( 5, cfg.trg(pulse))       =    startTime - triggerTime;      % estimated presentation delay after pulse
        timingz( 6, cfg.trg(pulse))       =    eststatTime;                  % the amount of presentation delay (only works if a presentation time is set)
    end

end

% get end of block indication
cfg.setup.B.sendTrigger(cfg.MEG.trigger.blockOffset)    % end of block trigger


%% SHUT DOWN AND SAVE DATA
save (fullfile( pwd, 'data', ppnum, [ppnum '-r' wrun '-localizer.mat']), 'responses', 'conditions', 'timingz', 'cfg', 'B');

end