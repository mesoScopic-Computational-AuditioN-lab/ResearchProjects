% cd 'C:\Users\jorie\AppData\Roaming\MathWorks\MATLAB Add-Ons\Toolboxes\Psychtoolbox-3\'
% SetupPsychtoolbox

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%     RUN: Tonotopy and pred_adapt_sound        %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear mex
addpath(genpath([pwd '/functions/'])); startup1;
% opacity = 0.9;
% PsychDebugWindowConfiguration([], opacity)

% script requires some toolboxes, check if they are installed
if ~license('test','Symbolic_Toolbox') || ~license('test','Statistics_Toolbox')
    LogicalStr = {'Not found', 'Installed'};
    error('u:stuffed:it', ['[!!!] Not all toolboxes are installed\n' ...
        '   -Symbolic Math Toolbox: ' LogicalStr{license('test','Symbolic_Toolbox')+1} '\n'...
        '   -Statistics and Machine Learning Toolbox: ' LogicalStr{license('test','Statistics_Toolbox')+1}])
end

%% enter participant info
while 1
    ppnum = input('ptc number: ','s');
    wrun  = input('run number: ','s');
    if ~exist( fullfile( pwd, 'data', ppnum, [ppnum '-illsize.mat'] ),'file')
        break
    else
        disp(['Participant ' ppnum ' already exists.']);
        whatnow = input('Assign new participant number (1) or overwrite participant data (2)?: ','s');
        if strcmp(whatnow,'2')
            break
        end
    end
end
if ~exist([pwd '/data/' ppnum],'dir'); mkdir([pwd '/data/' ppnum]); mkdir([pwd '/stimuli/' ppnum]); mkdir([pwd '/loudness/' ppnum]); end
setup = input('Environment: (0=Windows, 1=MacBook, 2=EyeLab): \n');
task  = input('Task: 1=LoudnessEqualisation, 2=GenTonesMain(implement), 3=PupilTask: \n');

%% Run experiment of choice
switch task

    case 1      % if we sellected loudness equalisation
        mainequalization(ppnum, wrun, setup);

        % display loudness
        load(fullfile( pwd, 'loudness', ppnum, [ppnum '-loudness.mat']),'all_loudness');
        vols = logistic_func(18, calc_logistic_growth(0.01, 18));
        disp(['Max volume is: ' num2str(vols(2,max(all_loudness)))]);

    case 2      % optional to precreate stimuli % fix plz
        mainpredstims(ppnum, wrun, setup);

    case 3
%         if str2double(wrun) > 12, ShowCursor; error('invalid run'), end
        mainpredsound_pupil(ppnum, wrun, setup, 1);
end
% savetempfile(ppnum,wrun)
% save_clean_timings(ppnum, stimlen, paddinglen)

%% shut down and clean up
ShowCursor;     ListenChar;     sca;   clearvars;