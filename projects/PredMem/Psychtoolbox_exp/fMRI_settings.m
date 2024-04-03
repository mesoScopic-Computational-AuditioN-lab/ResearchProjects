%% MAIN SETTINGS

addpath(genpath([pwd '/functions/']));          % add function to path


%% PARAMETERS

% sets
design.spr              = 2;            % number of sets per run

% set rest period timings
design.wait_Onset       = 10;           % waittime before stimulus onset (s)
design.wait_Offset      = 20;           % waittime from last stimulus to end of run (s)
design.wait_IBI         = 20;           % waittime of inter block interval (s)

% tr settings
design.TR               = 1.0;          % TR length

% convert waiting times to TRs and ceil to get upper bound
design.waitTR_Onset     = ceil(design.wait_Onset/design.TR);
design.waitTR_Offset    = ceil(design.wait_Offset/design.TR);
design.waitTR_IBI       = ceil(design.wait_IBI/design.TR);

% calculate lengths
design.blockLength      = (design.stimPerBlock * design.seqDur) / 1000;     % length of one block in sec
design.TRblockLength    = design.blockLength / design.TR;                   % length of one block in nr of TRs (can be decimals)
design.fullLength       = (design.spr * design.blockLength) + ...           % full length of one run in sec
                          ((design.spr - 1) * design.wait_IBI) + ...
                          design.wait_Onset + design.wait_Offset;
design.TRfullLength     = ceil(design.spr * design.TRblockLength) + ...    % full length of one run in nr of TRs
                          ((design.spr - 1) * design.waitTR_IBI) + ...
                          design.waitTR_Onset + design.waitTR_Offset;

% plan timings
design.loadbuffer       = 2;            % n-minus, before playback in nr of TRs
design.planplayback     = 1;            % n-minus, before playback to plan actual playback (schedule one TR in advance to garantee timings)

%% CREATE FOLDERS AND APPEND INFORMATION TO SETS AND DESIGN

% prep directory for saving timings
if ~exist('../DATA/fMRI', 'dir'); mkdir('../DATA/fMRI/'); end
if ~exist(sprintf('../DATA/Fmri/Subject_%s', ppnum), 'dir'); mkdir(sprintf('../DATA/Fmri/Subject_%s/', ppnum)); end

% save timingDir & fmriDir for documentation
design.timingDir = sprintf('../DATA/Fmri/Subject_%s/', num2str(design.subnum));   
design.fmriDir = ['../DATA/Subject_' num2str(design.subnum) '/'];

% load sets
set1 = load([design.fmriDir 'design_sub' num2str(design.subnum) '_set1.mat']);
set2 = load([design.fmriDir 'design_sub' num2str(design.subnum) '_set2.mat']);

% change ID to regr of set 2 (from 5 to 10)
regr = set1.regr;
for i = 1:5
    set2.regr(i).setID = i+5;
    regr(i+5).sequence = set2.regr(i).sequence;
    regr(i+5).setID = set2.regr(i).setID;
end

%% MONITOR SETTINGS
switch design.system

    case 0          % windows pc
        % screen info
        design.setup.disp_dist       = 500;                          % display distance in mm
        design.setup.screen_number   = max(Screen('Screens'));       % always use 2nd screen (or only monitor)
        [design.setup.screen_width, design.setup.screen_height] = Screen('DisplaySize', design.setup.screen_number);
        design.setup.hz              = round(Screen('FrameRate', design.setup.screen_number));
        design.setup.full_screen     = design.fullScreen;     

        % path and invironment info
        design.setup.base_path       = pwd;
        design.setup.environment     = 'Windows PC';
        Screen( 'Preference', 'SkipSyncTests', 1);      % skip synctest for testing

    case 1          % macbook
        % screen info
        design.setup.disp_dist       = 500;                          % display distance in mm
        design.setup.screen_height   = 180;   % mm
        design.setup.screen_width    = 285;   % mm
        design.setup.hz              = 60;
        design.setup.screen_number   = 0;
        design.setup.full_screen     = design.fullScreen;      

        % path and invironment info
        design.setup.base_path       = pwd;
        design.setup.environment     = 'Macbook';
        Screen( 'Preference', 'SkipSyncTests', 1);

    case 2          % fMRI
        % screen info
        design.setup.disp_dist       = 500;                          % display distance in mm
        design.setup.screen_number   = 1;
        [design.setup.screen_width, design.setup.screen_height] = Screen('DisplaySize', design.setup.screen_number);
        design.setup.hz              = round(Screen('FrameRate', design.setup.screen_number));
        design.setup.full_screen     = design.fullScreen; 

        % path and invironment info
        design.setup.base_path       = pwd;
        design.setup.environment     = 'fMRI';
        Screen( 'Preference', 'SkipSyncTests', 1);  % beamer does not support fliptest

end
% other screen settings
design.setup.dispsize    = [design.setup.screen_width design.setup.screen_height];
design.setup.pixsize     = Screen('Rect', design.setup.screen_number);
design.setup.w_px=design.setup.pixsize(3); design.setup.h_px=design.setup.pixsize(4);


%% KEY/BUTTON/PULSE SETTINGS
if IsWin        % if we are using windows
    design.keys.pulse           = 53;
    design.keys.shiftkey        = [160 161]; % leftshift rightshift
    design.keys.esckey          = 27;
    design.keys.space           = 32;
    
    design.visual.textfont      = 'Calibri';
elseif IsOSX    % if on MAC
    design.keys.pulse           = 93;
    design.keys.shiftkey        = [225 229]; % done
    design.keys.esckey          = 41;
    design.keys.space           = 44;

    design.visual.textfont      = 'Arial';
end


%% VISUAL SETTINGS
% misc. settings
design.visual.c3              = ones(1,3)*255;                      % set color white
design.visual.backgr          = round(0.5*design.visual.c3);        % set background color (0=black, 1=white, .5=meangrey)
design.visual.textcol         = design.visual.c3*1;                 % set text color
design.visual.fontsize        = 42;                                 % font size in pixels

% fixation bull parameters
design.visual.fixfrac             = 2;
design.visual.bull_dim_fact       = 1;              % dim factor of bullseye fixation
design.visual.bull_dim_col        = [0.7 0.7 0.7];    
design.visual.bull_eye_col        = [0 0 0];
design.visual.bull_in_col         = [1 1 1];
design.visual.bull_in_col_cor     = [0 1 0];
design.visual.bull_in_col_inc     = [1 0 0];
design.visual.bull_out_col        = [0 0 0];
design.visual.bull_fixrads        = design.visual.fixfrac * [44 20 12];        % midpoint, inner_ring, outer_ring size

% degree to pixel
design.visual.bullsize               = ang2pix( design.visual.bull_dim_fact, design.setup.disp_dist, design.setup.screen_width, design.setup.pixsize(3),1);

%% SOUND SETTINGS
% audio driver settings
design.sound.nrchannels          = 1;                    % sellect number of channels (1=mono-audio, 2=stereo)
design.sound.samp_rate           = design.fs;            % sampling rate used (must match audio driver samp-rate, e.g. 44100 or 48000)
design.sound.max_latancy         = [];                   % define max latancy, note that to short will lead to problemns in presenting (empty lets psychtoolbox decide)

%% INSTRUCTION SETTINGS
design.visual.waitforscan        = {'Waiting for scanner...'};
design.visual.waitforspace       = {'Run complete, press space to close screen...'};

%% LOAD PREGENERATED DATA

% loop over all sets in block and load them all
blockAudio_sets = {};
for b = 1:design.spr
    % load pregenerated block audio sets ontsizeor current run
    [blockAudio,Fs]         = audioread([design.resultsDir sprintf('sub%s_run%s_set%d.wav', ppnum, wrun, b)]);
    blockAudio_sets{b}      = blockAudio.';

    % adjust audio from mono to stereo if needed and desired
    if size(blockAudio_sets{b}, 2) < design.sound.nrchannels; blockAudio_sets{b} = [blockAudio; blockAudio]; end

    % check if samplerate is correct
    if Fs ~= design.sound.samp_rate
        error('Sample rate of WAV file does not match playback speed, please regenerate at correct sampling rate')
    end

    clear blockAudio;
end

%% OPEN SCREEN
% open screen and set transparanct mode
[w, screenrect]  =   Screen( 'OpenWindow', design.setup.screen_number, design.visual.backgr);
design.setup.screenrect = screenrect;
Screen( w, 'BlendFunction', GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
Screen( 'FillRect', w, design.visual.backgr);  Screen( 'Flip', w);

% test screen refreshrate
itis        = 75;
x           = NaN(1, itis+1);
for i = 1:itis+1
    x(i)        = Screen('Flip', w);
end
x(1)=[];
design.setup.estirate    = 1/mean(diff(x));
if design.setup.estirate < (design.setup.hz-3) || design.setup.estirate > (design.setup.hz+3)
    sca; ShowCursor;
    error('[!!!] Refresh rate estimated at %g Hz instead of %g Hz',design.setup.estirate, design.setup.hz);
end

%% SAVE SETTINGS
save (fullfile( [design.timingDir sprintf('_sub%s_run%s_settings.mat', ppnum, wrun)]))

%% CLEAN UP AND DECLUTTER
clear ans; clear checkvers; clear designFn; clear i; clear x; clear b; clear oldEnableFlag;
