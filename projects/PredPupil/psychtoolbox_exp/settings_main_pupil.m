%% MAIN SETTINGS - pupillometry

cfg                 = [];                   % setup struct 
addpath(genpath([pwd '/functions/']));      % add functions to path
cfg.task            = 1;                    % passive task or orthogonal task

%% COUNTERBALANCING SETTINGS
% primary conditions
cfg.RAND_sampling_prob          = 0.5;                          % distribution probability for RAND segments
cfg.REG_sampling_prob           = 0.9;                          % distribution probability for REG segments
cfg.RAND_sampling_width         = 2;                          % distribution width for RAND segments
cfg.REG_sampling_width          = 0.2;                          % distribution width for REG segments

% primary conditions labels
cfg.mainConditions_pre_descr    =   {'REG', 'RAND'};            % pre-boundary conditions
cfg.mainConditions_post_descr   =   {'REG', 'RAND', 'd'};       % post-boundary conditions (d: for dREG, or dRAND relying on pre)
cfg.tpc                         =   24;                         % number trials per conditions (must be divisable by nr secondary condtions)

% second conditions
cfg.distWidths                  =   [-0.15 0 0.15];               % diviation from RAND or REG sampling width
cfg.distProbs                   =   [-0.05 0 0.05];             % diviation from RAND or REG probability sampling

% other non-counterbalanced conditions
cfg.onsetJitter                 =   [-5 0 5];                   % number of tones to jittered onset

% trial settings
cfg.n_segments      = 2;                    % number of segments per block
cfg.n_blocks        = 144;                  % number of blocks
cfg.n_runs          = 16;                   % number of runs
cfg.n_distr         = 2;                    % number of distributions
cfg.n_freq          = 2;                    % number of center frequencies
cfg.n_probs         = cfg.n_segments*cfg.n_blocks;%15;% number probabilities
cfg.tps             = [20 40];   % trails in segments without jitter

cfg.bpr             = cfg.n_blocks/cfg.n_runs;%4;% how many blocks per run

% ommision settings
cfg.opb             = [0];                  % ommision trails per segment per condition 
cfg.n_ommis_bl      = 1;                    % number of ommission conditions

% block settings
cfg.n_freq_bl       = 8;                    % total number of center frequencys over blocks
cfg.n_surp_bl       = 0;                    % number of surprise conditions over blocks


%% STIMULUS SETTINGS
% stimulus settings
cfg.min_prob        = 0.05;                                                                         % set miniminum probability 
cfg.stim_probs      = logistic_func(cfg.n_probs, calc_logistic_growth(cfg.min_prob, cfg.n_probs));  % calculate stim probs (option for no symb math toolbox?)
cfg.minfreq         = 500;                                                                          % minimum center frequency to use
cfg.maxfreq         = 3000;                                                                         % maximum center frequency to use
cfg.numsteps        = 8;                                                                            % number of center frequencies
cfg.cent_freqs      = linspace(log2(cfg.minfreq), log2(cfg.maxfreq), cfg.numsteps);                 % center frequencies (n_distr)
cfg.block_pairs     = [1 2 3; 5 6 7]; %[1 2 3 4; 5 6 7 8];                                          % block pairings
cfg.dist_widths     = [1/3, 2/3];                                                                   % width possibilities of distributions
cfg.dist_width_block = ones(1,cfg.n_blocks);
for blk = 1:cfg.n_blocks                                                                            % go over blocks and set width of distribution
    cfg.dist_width_block(blk) = cfg.dist_widths(randi(length(cfg.dist_widths)));
end
cfg.oct_width       = ones(1,cfg.numsteps) * 1/3;                                               % width of distribution (n_distr), alter to adjust sampling
cfg.oct_res         = 12;                                                                       % set a octive resolution for sampling (e.g. 12=1/12 oct steps) or set to 'none'
% cfg.loudLevel       = [0.01 .02 .05 .1 .2 .25 .33 .5 0.7      5 0.9 1];                         % set loudness levels
cfg.loudLevel       = logistic_func(18, calc_logistic_growth(0.01, 18));                        % set loudness levels
cfg.loudLevel       = cfg.loudLevel(2,:);

% timings
cfg.stim_t          = 0.2;                  % stimulus presentation length (in sec)
cfg.iti             = 0.05;                 % inter trial interval
cfg.ibi             = 5;                    % inter block interval (was 11.2)
cfg.isi             = 0;                    % inter segment interval
cfg.ramp_ops        = 0.005;                % rampup of stimuli 
cfg.ramp_dns        = 0.005;                % rampdown of stimuli
cfg.padding         = 0.002;                % amount of padding added to stimuli (50% before, 50% after) 0.01/0.005
cfg.countd          = .8;                   % countdown speed for new block (not used in mri)

% calculate lengths
cfg.len_segment     = (cfg.tps(2) * (cfg.stim_t + cfg.iti));                            % length of a segment
cfg.len_block       = (cfg.len_segment * (cfg.n_segments + cfg.isi)) - cfg.isi;         % length of block

% audioplayback settings
cfg.wait_onset  = 3;                        % silence time before first trial in sec  
cfg.wait_offsett= 3;                        % silence time after first trial in sec
cfg.pressdelay      = 1;                    % presentation delay to garantee audio timing
cfg.playbacklength  = 1;                    % how many stimuli to playback as '1' - div by tps
                                            % (higher number limits timing data but reduces timing caused issues)
% trigger settings
cfg.trigger.stimOnset       = 100;      % trigger: start stim
cfg.trigger.blockOnset      = 50;       % trigger: start new block (11 to (10+nrblocks)) 
cfg.trigger.blockOffset     = 70;       % trigger: start new block (11 to (10+nrblocks)) 
cfg.trigger.firstSegOnset   = 20;       % trigger: first segment start
cfg.trigger.secSegOnset     = 21;       % trigger: second segment start

%% VISUAL SETTINGS
% misc. settings
cfg.visual.c3              = ones(1,3)*255;        % set color white
cfg.visual.backgr          = round(0.5*cfg.visual.c3);        % set background color (0=black, 1=white, .5=meangrey)
cfg.visual.textcol         = cfg.visual.c3*1;                 % set text color
cfg.visual.fontsize        = 36;                   % font size in pixels

% fixation bull parameters
cfg.visual.fixfrac             = 2;
cfg.visual.bull_dim_fact       = 1;              % dim factor of bullseye fixation
cfg.visual.bull_dim_col        = [0.7 0.7 0.7];    
cfg.visual.bull_eye_col        = [0 0 0];
cfg.visual.bull_in_col         = [1 1 1];
cfg.visual.bull_in_col_cor     = [0 1 0];
cfg.visual.bull_in_col_inc     = [1 0 0];
cfg.visual.bull_out_col        = [0 0 0];
cfg.visual.bull_fixrads        = cfg.visual.fixfrac * [44 20 12];        % midpoint, inner_ring, outer_ring size

% display warning if stimulus settings do not sum to 1
if mean(sum(cfg.stim_probs)) < 1
    warning('Probabilities in `stim_probs` do not sum to 1; [nr: %s] - please check row 29 of settings', num2str(find(sum(stim_probs) < 1)))
end

%% SOUND SETTINGS
% audio driver settings
cfg.sound.nrchannels      = 1;                    % sellect number of channels (1=mono-audio)
cfg.sound.samp_rate       = 48000;                % sampling rate used (must match audio driver samp-rate, e.g. 44100 or 48000)
cfg.sound.max_latancy     = [];                   % define max latancy, note that to short will lead to problemns in presenting (empty lets psychtoolbox decide)

%% SET DIRECTORY
dirout          = [pwd '/data/' ppnum];

%% MONITOR SETTINGS
switch setup

    case 0          % windows pc
        % screen info
        cfg.setup.disp_dist       = 500;                          % display distance in mm
        cfg.setup.screen_number   = max(Screen('Screens'));       % always use 2nd screen (or only monitor)
        [cfg.setup.screen_width, cfg.setup.screen_height] = Screen('DisplaySize', cfg.setup.screen_number);
        cfg.setup.hz              = round(Screen('FrameRate', cfg.setup.screen_number));
        cfg.setup.full_screen     = 1;    

        % path and invironment info
        cfg.setup.base_path       = pwd;
        cfg.setup.environment     = 'Windows PC';
        Screen( 'Preference', 'SkipSyncTests', 1);      % skip synctest for testing

    case 1          % macbook
        % screen info
        cfg.setup.disp_dist       = 500;                          % display distance in mm
        cfg.setup.screen_height   = 180;   % mm
        cfg.setup.screen_width    = 285;   % mm
        cfg.setup.hz              = 60;
        cfg.setup.screen_number   = 0;
        cfg.setup.full_screen     = 1;      

        % path and invironment info
        cfg.setup.base_path       = pwd;
        cfg.setup.environment     = 'Macbook';
        Screen( 'Preference', 'SkipSyncTests', 1);

    case 2          % Eye-Lab
        % screen info
        cfg.setup.disp_dist       = 550;                          % display distance in mm
        cfg.setup.screen_number   = 0;       % always use 2nd screen (or only monitor)
        [cfg.setup.screen_width, cfg.setup.screen_height] = Screen('DisplaySize', cfg.setup.screen_number);
        cfg.setup.hz              = round(Screen('FrameRate', cfg.setup.screen_number));
        cfg.setup.full_screen     = 1;                            % 

        % path and invironment info
        cfg.setup.base_path       = pwd;
        cfg.setup.environment     = 'Eye-Lab';
        Screen( 'Preference', 'SkipSyncTests', 1);
end
% other screen settings
cfg.setup.dispsize    = [cfg.setup.screen_width cfg.setup.screen_height];
cfg.setup.pixsize     = Screen('Rect', cfg.setup.screen_number);
cfg.setup.w_px=cfg.setup.pixsize(3); cfg.setup.h_px=cfg.setup.pixsize(4);

%% INSTRUCTION SETTINGS

cfg.visual.tskexp              = {'Keep your eyes on the fixation dot, you dont have to do anything'};
cfg.visual.waitforspace        = {'Run complete, press space to close screen...'};
cfg.visual.waitforstartexp     = {'Keep your eyes on the fixation dot, you dont have to do anything', ' ', 'Press any button to continue'};
cfg.visual.waitforstartblock   = {'Press any button to start the block'};
cfg.visual.endofblock          = {'End of block, you may take a break', ' ', 'Call the researcher when you are ready to start again'};
cfg.visual.recalibration       = {'Calibration', ' ', 'Will recalibrate shortly ...'};

%% KEY/BUTTON/PULSE SETTINGS

if setup == 3   % if MEG
    cfg.keys.next            = [32 13 49 50 51 52 53 54 55 56 57 58];
    cfg.keys.pulse           = 53;

    cfg.keys.shiftkey        = [160 161]; % leftshift rightshift
    cfg.keys.rhk             = [37 40 39 38];  % LDRU
    cfg.keys.lhk             = [65 83 68 87];  % asdw
    cfg.keys.esckey          = 27;
    cfg.keys.space           = 32;
    
    cfg.visual.textfont        = 'Calibri';
elseif IsWin     % else we can use serial input key 5 for triggers
    cfg.keys.next            = [32 13 49 50 51 52 53 54 55 56 57 58];
    cfg.keys.pulse           = 53;

    cfg.keys.shiftkey        = [160 161]; % leftshift rightshift
    cfg.keys.rhk             = [37 40 39 38];  % LDRU
    cfg.keys.lhk             = [65 83 68 87];  % asdw
    cfg.keys.esckey          = 27;
    cfg.keys.space           = 32;
    
    cfg.visual.textfont        = 'Calibri';
elseif IsOSX
    cfg.keys.next            = 30;
    cfg.keys.pulse           = 34;

    cfg.keys.shiftkey        = [225 229]; % done
    cfg.keys.rhk             = [80 81 79 82];  % LDRU (????)
    cfg.keys.lhk             = [4 22 7 26];    % asdw
    
    cfg.keys.esckey          = 41;
    cfg.keys.space           = 44;
    cfg.visual.textfont        = 'Arial';
end

%% OPEN SCREEN

% open screen and set transparanct mode
[w, screenrect]  =   Screen( 'OpenWindow', cfg.setup.screen_number, cfg.visual.backgr);
cfg.setup.screenrect = screenrect;
Screen( w, 'BlendFunction', GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
% [a1 a2 a3] = Screen( 'ColorRange', w, 1);%[maximumvalue],[clampcolors], [applyToDoubleInputMakeTexture]);
Screen( 'FillRect', w, cfg.visual.backgr);  Screen( 'Flip', w);

% test screen refreshrate
itis        = 75;
x           = NaN(1, itis+1);
for i = 1:itis+1
    x(i)        = Screen('Flip', w);
end
x(1)=[];
cfg.setup.estirate    = 1/mean(diff(x));
if setup~=0
    if cfg.setup.estirate < (cfg.setup.hz-3) || cfg.setup.estirate > (cfg.setup.hz+3) % ruime marge nog! 
        sca; ShowCursor;
        error('[!!!] Refresh rate estimated at %g Hz instead of %g Hz',cfg.setup.estirate, cfg.setup.hz);
    end
end

%% TRANSFORM VARIABLES INTO PIXEL DIMENSION (FROM DEGREE)

% degree to pixel
cfg.visual.bullsize        = ang2pix( cfg.visual.bull_dim_fact, cfg.setup.disp_dist, cfg.setup.screen_width, cfg.setup.pixsize(3),1);

%% SAVE SETTINGS
save (fullfile( pwd, 'data', ppnum, [ppnum '_settings.mat']));

%% Clean up and declutter
clear i; clear itis; clear x;