%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%     RUN: Tonotopy and pred_adapt_sound        %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear mex
addpath(genpath([pwd '/functions/'])); startup1;

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
setup = input('Environment: (0=Windows, 1=MacBook, 2=fMRI, 3=MEG): \n');
task  = input('Task: (1=Tonotopy, 2=MainTask, 3=LoudnessEqualisation, 4=GenTonesMain, 5=MegTask, 6=LeftRightEqualisation): \n');

%% Run experiment of choice
switch task

    case 1      % if we sellected Tonotopy
        maintonotopy(ppnum, wrun, setup);       % run takes 6 min (8 min)

    case 2      % if we sellected Main Task
        if str2double(wrun) > 12, ShowCursor; error('invalid run'), end
        mainpredsound(ppnum, wrun, setup);      % run takes 7 min

    case 3      % if we sellected loudness equalisation
        leftrightequalization(ppnum, wrun, setup);
        mainequalization(ppnum, wrun, setup);

        % display loudness
        load(fullfile( pwd, 'loudness', ppnum, [ppnum '-loudness.mat']),'all_loudness');
        vols = logistic_func(18, calc_logistic_growth(0.01, 18));
        disp(['Max volume is: ' num2str(vols(2,max(all_loudness)))]);

    case 4      % optional to precreate stimuli % fix plz
        mainpredstims(ppnum, wrun, setup);

    case 5      % if we sellect MEG mainpred
        mainpredsound_MEG(ppnum, wrun, setup)

    case 6
        leftrightequalization(ppnum, wrun, setup);
end
% savetempfile(ppnum,wrun)
% save_clean_timings(ppnum, stimlen, paddinglen)

%% shut down and clean up
ShowCursor;     ListenChar;     sca;   clearvars;