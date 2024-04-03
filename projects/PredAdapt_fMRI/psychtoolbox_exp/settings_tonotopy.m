%% MAIN SETTINGS

cfg = [];                   % set up struct

% frequency settings
cfg.minfreq         = 200;      % minimum frequency in hz
cfg.maxfreq         = 6000;     % maximum frequency in hz
cfg.amplitude_mod   = 6;        % amplitude modulation in hz
cfg.mod_index       = 1;        % depth of amplitude modulation
cfg.mod_min         = 0;        % minimum of amplitude modulation
cfg.mod_phase       = 'random'; % set phase in degrees (0-360) or to random
% cfg.loudLevel       = [0.01 .02 .05 .1 .2 .25 .33 .5 0.75 0.9 1];   % set loudness levels
% cfg.loudLevel       = logistic_func(18, calc_logistic_growth(0.01, 18));    % set loudness levels
% cfg.loudLevel       = cfg.loudLevel(2,:);
cfg.loudLevel       = logspace(-2, 0, 18);

% ramp settings
cfg.rampup_dur      = 0.015;     % ramp-up duration in seconds
cfg.rampdown_dur    = 0.015;     % ramp-down duration in seconds

% repetition settings
cfg.tpb             = 80;       % presented frequencies in block
cfg.n_blocks        = 3;        % number of blocks
cfg.num_freqs       = 240;      % number of frequencies % can be lower
cfg.min_leap        = 1.0;      % minimum leap between stimulations in octives
cfg.temp_leap       = 2;        % steps to not use frequencies in similair range
cfg.per_gap         = 0.05;     % percentage silent gaps
cfg.n_gaps          = cfg.tpb*cfg.per_gap;   % number of gaps within a block

% timing settings
cfg.TR              = 1.8;      % TR in sec
cfg.stim_len        = 1.4;      % stimulus presentation length in sec
cfg.isi             = 0.4;      % inter stimulus interval in sec
cfg.ibiTR           = 8;        % number of TRs silence between blocks
cfg.waitTR_onset    = 5;        % number of TRs silence before first trial (12.6 sec)
cfg.waitTR_offset   = 5;        % number of TRs silence after last trial
cfg.ibi             = cfg.ibiTR*cfg.TR;          % inter block interval (silence interval) in sec
cfg.wait_onset      = cfg.waitTR_onset*cfg.TR;   % silence time before first trial in sec
cfg.wait_offset     = cfg.waitTR_offset*cfg.TR;  % silence time after first trial in sec
cfg.presdelay       = 0;    % time delay to present after pulse trigger (0ms here)

addpath(genpath([pwd '/functions/']));  % add functions to path
% check if number of silence gaps is possible
if rem(cfg.tpb*cfg.per_gap, 1) > 0
    error(['[!!!] ERROR: Failed to take ' num2str(cfg.per_gap*100) '% of ' num2str(cfg.tpb) ' trials (trials-per-blocks)']);
end


%% CALCULATE STIMULI ORDER
cfg.freq_array  = linspace(log2(cfg.minfreq), log2(cfg.maxfreq), cfg.num_freqs);     % get freq array in equal (octv) steps
cfg.leap        = ceil(cfg.min_leap / diff(cfg.freq_array([1 2])));                  % get leap distance in number of steps
cfg.stim_order  = quasirandom_sequence(cfg.num_freqs, cfg.leap, cfg.temp_leap);      % then calculate quasirandom_sequence order
cfg.stims       = cfg.freq_array(cfg.stim_order);

% decide silent gap location
cfg.gaps_loc    = nan(1, cfg.tpb*cfg.n_blocks*cfg.per_gap);   
for block = 0:cfg.n_blocks-1                                            
    cfg.gaps_loc(1+(block*cfg.n_gaps):(block+1)*cfg.n_gaps)     = sort(randperm(cfg.tpb-2, cfg.n_gaps)+1+(block*cfg.tpb));
end

% add silence gaps to stim order
cfg.gaps_loc = cfg.gaps_loc + linspace(1, cfg.tpb*cfg.n_blocks*cfg.per_gap, cfg.tpb*cfg.n_blocks*cfg.per_gap)-1;  % deal with shifting indexing problem
for gap = 1:cfg.tpb*cfg.n_blocks*cfg.per_gap
    cfg.stims      = [cfg.stims(1:cfg.gaps_loc(gap)-1) 0 cfg.stims(cfg.gaps_loc(gap):end)];
    cfg.stim_order = [cfg.stim_order(1:cfg.gaps_loc(gap)-1) 0 cfg.stim_order(cfg.gaps_loc(gap):end)];
end

cfg.tpb = cfg.tpb+cfg.n_gaps;       % set true number of tpb

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

    case 2          % fMRI
        % screen info
        cfg.setup.disp_dist       = 500;                          % display distance in mm
        cfg.setup.screen_number   = 1;       % always use 2nd screen (or only monitor)
        [cfg.setup.screen_width, cfg.setup.screen_height] = Screen('DisplaySize', cfg.setup.screen_number);
        cfg.setup.hz              = round(Screen('FrameRate', cfg.setup.screen_number));
        cfg.setup.full_screen     = 1;                            % 

        % path and invironment info
        cfg.setup.base_path       = pwd;
        cfg.setup.environment     = 'fMRI';

    case 3          % MEG (under construction)
        % screen info
        cfg.setup.disp_dist       = 500;                          % display distance in mm
        cfg.setup.screen_number   = max(Screen('Screens'));       % always use 2nd screen (or only monitor)
        [cfg.setup.screen_width, cfg.setup.screen_height] = Screen('DisplaySize', cfg.setup.screen_number);
        cfg.setup.hz              = round(Screen('FrameRate', cfg.setup.screen_number));
        cfg.setup.full_screen     = 1;                            % 

        % path and invironment info
        cfg.setup.base_path       = pwd;
        cfg.setup.environment     = 'MEG';
end
% other screen settings
cfg.setup.dispsize    = [cfg.setup.screen_width cfg.setup.screen_height];
cfg.setup.pixsize     = Screen('Rect', cfg.setup.screen_number);
cfg.setup.w_px=cfg.setup.pixsize(3); cfg.setup.h_px=cfg.setup.pixsize(4);


%% KEY/BUTTON/PULSE SETTINGS

if setup == 3   % if MEG
    %https://intranet.donders.ru.nl/index.php?id=lab-response-fiberoptic&no_cache=1
    %do we want to use 2 pads or one? e.g. index finger both hands
    cfg.keys.fopad           = [97 98 99 100 101 102 103 104]; 
    cfg.keys.esckey          = 27;
    cfg.keys.space           = [32 97 98 99 100 101 102 103 104];
    
    cfg.keys.keycomb1        = 112; % f1
    cfg.keys.keycomb2        = 71;  % g

    cfg.visual.textfont        = 'Calibri';
elseif IsWin     % else we can use serial input key 5 for triggers
    cfg.keys.pulse           = 53;

    cfg.keys.shiftkey        = [160 161]; % leftshift rightshift
    cfg.keys.rhk             = [37 40 39 38];  % LDRU
    cfg.keys.lhk             = [65 83 68 87];  % asdw
    cfg.keys.esckey          = 27;
    cfg.keys.space           = 32;
    
    cfg.visual.textfont        = 'Calibri';
elseif IsOSX
    cfg.keys.pulse           = 93;

    cfg.keys.shiftkey        = [225 229]; % done
    cfg.keys.rhk             = [80 81 79 82];  % LDRU (????)
    cfg.keys.lhk             = [4 22 7 26];    % asdw
    
    cfg.keys.esckey          = 41;
    cfg.keys.space           = 44;
    cfg.visual.textfont        = 'Arial';
end

%% SOUND SETTINGS
% audio driver settings
cfg.sound.nrchannels      = 1;                    % sellect number of channels (1=mono-audio)
cfg.sound.samp_rate       = 48000;                % sampling rate used (must match audio driver samp-rate, e.g. 44100 or 48000)
cfg.sound.max_latancy     = [];                   % define max latancy, note that to short will lead to problemns in presenting, empty let psychtoolbox decide

%% INSTRUCTION SETTINGS

cfg.visual.tskexp              = {'Keep your eyes on the fixation dot, you dont have to do anything'};
cfg.visual.waitforscan         = {'Waiting for scanner...'};


%% VISUAL SETTINGS
% misc. settings
cfg.visual.c3              = ones(1,3)*255;        % set color white
cfg.visual.backgr          = round(0.5*cfg.visual.c3);        % set background color (0=black, 1=white, .5=meangrey)
cfg.visual.textcol         = cfg.visual.c3*1;                 % set text color
cfg.visual.fontsize        = 36;                   % font size in pixels

% fixation bull parameters
cfg.visual.fixfrac             = 1;
cfg.visual.bull_dim_fact       = 1;              % dim factor of bullseye fixation
cfg.visual.bull_dim_col        = [0.7 0.7 0.7];    
cfg.visual.bull_eye_col        = [0 0 0];
cfg.visual.bull_in_col         = [1 1 1];
cfg.visual.bull_in_col_cor     = [0 1 0];
cfg.visual.bull_in_col_inc     = [1 0 0];
cfg.visual.bull_out_col        = [0 0 0];
cfg.visual.bull_fixrads        = cfg.visual.fixfrac * [44 20 12];        % midpoint, inner_ring, outer_ring size


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
save (fullfile( pwd, 'data', ppnum, [ppnum '_settings_tonotopy.mat']));

%% Clean up and declutter
clear gap; clear i; clear itis; clear x;