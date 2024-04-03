function fMRI_ExpMain(design,ppnum,wrun)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Fast access for testing                                                 %
%                                                                         %
% addpath(genpath([pwd '/functions/']));startup1;                         %
% Screen( 'Preference', 'SkipSyncTests', 1);                              %
% ppnum ='1'; wrun ='1'; setup =0;     % set to 2 for fMRI                %
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fMRI_settings;
addpath(genpath([pwd '/functions/']));      % add functions to path

%% PULSE-WISE TIMING PREP

pulses              = zeros( 2, design.TRfullLength);
% loop over all sets planning values
for b = 1:design.spr
    % calculate the blockonset time
    blockonset      =   design.waitTR_Onset + ...
                        design.waitTR_IBI * (b-1) + ...
                        ceil(design.TRblockLength * (b-1)) + 1; 

    % sellect pulses to load audio into buffer
    pulses( 1, blockonset - design.loadbuffer)      = b;    % [1]: sellect at which pulse to load buffer
    pulses( 2, blockonset - design.planplayback)    = b;    % [2]: sellect at which pulse to plan playback
end


%% SAVE TIMINGS

% define timing matrix and start populating where possible
timings                 = nan( 10, design.TRfullLength);
timings( 1, :)          = nan;                              % [1]: Trial number in set [from infomat]
timings( 2, :)          = nan;                              % [2]: Condition identifier [from infomat]
timings( 3, :)          = nan;                              % [3]: Set ID [from infomat]
timings( 4, :)          = nan;                              % [4]: Relative onset time of miniblock to start of block [from infomat]
timings( 5, :)          = linspace(0,design.fullLength-1, ...
                                    design.TRfullLength);   % [5]: Relative pulse time to first pulse
timings( 6, :)          = str2double(wrun);                 % [6]: What Run
timings( 7, :)          = nan;                              % [7]: Set ID in run
timings( 8, :)          = nan;                              % [8]: Playback start positions (boolean)
timings( 9, :)          = nan;                              % [9]: Actual pulse timestamp
timings( 10, :)         = nan;                              % [10]: Real audio playback Onset


% loop over all sets planning values and append/copy infomat files to timings matrix
for b = 1:design.spr

    % load information matrix
    load([design.resultsDir sprintf('sub%s_run%s_set%d.mat', ppnum, wrun, b)],'infomat');
    fprintf('Loading: sub%s_run%s_set%d.mat\n', ppnum, wrun, b);

    % calculate the blockonset time
    blockonset      =   design.waitTR_Onset + ...
                        design.waitTR_IBI * (b-1) + ...
                        ceil(design.TRblockLength * (b-1)) + 1; 
    blockoffset     =   blockonset + design.TRblockLength - 1;

    % save information in timings matrix
    timings( 1, blockonset:blockoffset)          = repelem(infomat(:, 1), design.TRblockLength / length(infomat));       % [1]: Trial number in set (from infomat)
    timings( 2, blockonset:blockoffset)          = repelem(infomat(:, 2), design.TRblockLength / length(infomat));       % [2]: Condition identifier (from infomat)
    timings( 3, blockonset:blockoffset)          = repelem(infomat(:, 3), design.TRblockLength / length(infomat));       % [3]: Set ID (from infomat)
    timings( 4, blockonset:blockoffset)          = repelem(infomat(:, 4), design.TRblockLength / length(infomat));       % [4]: Relative onset time of miniblock (from infomat)
    timings( 7, blockonset:blockoffset)          = b;
    timings( 8, blockonset)                      = 1; 

    % cleanup clutter
    clear infomat
end

% load fixation bull into memory
BullTex(1)     = get_bull_tex_2(w, design.visual.bull_eye_col, design.visual.bull_in_col, design.visual.bull_out_col, design.visual.bull_fixrads);
BullTex(2)     = get_bull_tex_2(w, design.visual.bull_eye_col, design.visual.bull_in_col_cor, design.visual.bull_out_col, design.visual.bull_fixrads);
BullTex(3)     = get_bull_tex_2(w, design.visual.bull_eye_col, design.visual.bull_in_col_inc, design.visual.bull_out_col, design.visual.bull_fixrads);

%% INITIALIZE
PsychPortAudio('Close'); % make sure is closed before initializing
InitializePsychSound(1);
pahandle = PsychPortAudio('Open', [], 1, 2, design.sound.samp_rate, design.sound.nrchannels, [], []);

% set keys
pulsekey            =   design.keys.pulse;       % key for pulse trigger (5)
esckey              =   design.keys.esckey;      % key for escape key
shiftkey            =   design.keys.shiftkey;    % key for shiftkey
space               =   design.keys.space;       % key for spacebar

% visualy
bullrect            =   CenterRect([0 0 design.visual.bullsize design.visual.bullsize], screenrect);

%% RUN TRIALS

% initialize responses
responses           = [];                        % for later expension

% present waiting for scanner message till first pulse
disptext(w, design.visual.waitforscan{1}, screenrect, ...
            design.visual.fontsize, design.visual.textfont, design.visual.textcol);
Screen('Flip', w);

% run over volumes of this block
for pulse = 1: length(pulses)
    
    % start waiting for triggers
    pulseTime = waitforpulse(pulsekey, esckey, shiftkey, design, pulses, timings, responses);  waitfornokey;
%     pulseTime = WaitSecs(design.TR*0.95);    % for testing

    % save timing
    timings( 9, pulse)              = pulseTime;                                    % pulse timestamp

    % draw fixation
    Screen('DrawTexture',w,BullTex(1),[],bullrect);
    [~] = Screen('Flip', w);

    % if pulse array 1 states prepbuffer - do so
    if pulses(1, pulse) > 0
        blockAudio                  = blockAudio_sets{pulses(1, pulse)};            % get block audio
        PsychPortAudio('FillBuffer', pahandle, blockAudio);                         % fill block audio buffer
        clear blockAudio;
    end

    % if pulse array 2 states plan playback - do so
    if pulses(2, pulse) > 0
        [eststatTime]               = PsychPortAudio('Start', pahandle, [], timings( 9, pulse) + design.TR );  
        timings( 10, pulse)         = eststatTime;                                  % save calculated audio onset timing
    end
end

% check if pulses were missed and notify of mismatch (in TR waiting times)
if max(diff(timings(9,:))) > design.TR * 1.5; warning('TR mismatch found, pulse missed? (check timing)'); end

%% SHUT DOWN AND SAVE DATA
save (fullfile( [design.timingDir sprintf('sub%s_run%s_timings.mat', ppnum, wrun)]), 'timings', 'pulses', 'responses');

% Wait untill the used presses space to close (so that the participant doesnt have to look at a windows screen while taking a break
disptext(w, design.visual.waitforspace{1}, screenrect, ...
            design.visual.fontsize, design.visual.textfont, design.visual.textcol);
Screen('Flip', w);
waitforpulse(space,  esckey, shiftkey, design, pulses, timings, responses);  waitfornokey;
PsychPortAudio('Close'); 

end