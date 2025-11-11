function all_loudness = leftrightequalization(ppnum, wrun, setup)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Fast acces for testing purpos                                                        %
%                                                                                       %
% addpath(genpath([pwd '/functions/']));addpath(genpath([pwd '/stimuli/']));startup1;   %
% Screen( 'Preference', 'SkipSyncTests', 1);                                            %
% ppnum ='1'; wrun ='1'; setup =2; % set to 2 for fMRI                                    %
%                                                                                       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% SETTINGS

% equalisation
settings_tonotopy;                                  % fetch some information from tonotopy settings
equal.refstim         =   6;                        % what stimulus to use as refference
equal.refloud         =   14;                        % what is the 'refference' loudness
equal.minfreq         =   cfg.minfreq;              % minimum frequency to equalize
equal.maxfreq         =   cfg.maxfreq;              % maxiumum frequency to equalize
equal.nfreq           =   9;                        % number of frequencies to use in equalisation
equal.freq0           =   2.^(linspace(log2(equal.minfreq),log2(equal.maxfreq),equal.nfreq));     % create logspace freq array

% sound
equal.sampdur         =   0.5;                      % duration of sample in ms
equal.sil_gap         =   0.25;                     % length of sillence gap
equal.intloudness     =   cfg.loudLevel;            % set loudness levels
equal.ampl            =   .95;                      % amplitude
equal.nrchannels      =   cfg.sound.nrchannels;     % number of channels used
equal.samp_rate       =   cfg.sound.samp_rate;      % sampling rate used

% amplitude mod
equal.amplmod         =   8;                        % frequency of amplitude modulation
equal.mod_index       =   1;                        % index of amplitude mod
equal.mod_min         =   0;                        % minimum of amplitude modulation

% misc
equal.fontsize        =   cfg.visual.fontsize;      % set fontsize
equal.backgr          =   cfg.visual.backgr;        % set background collor
if IsWin 
    equal.textfont    = 'Calibri';                  % set fontfamily 
elseif IsOSX
    equal.textfont    = 'Arial';
end
equal.textcol         =   cfg.visual.c3*1;          % set text color
equal.introtxt        =   {'Judge the loudness of the sounds', ' ', '(Press any key to continue)'};
equal.keytxt          =   {'First sound louder: 1', 'Second sound louder: 2', ...
                           'Loudness about the same: 3', ' ','(Press any key to start)'};
equal.bull_eye_col    =   cfg.visual.bull_eye_col;  % color of bullseye
equal.bull_in_col     =   cfg.visual.bull_in_col;   % color of inner ring
equal.bull_out_col    =   cfg.visual.bull_out_col;  % color of outer ring

equal.bull_fixrads    =   cfg.visual.bull_fixrads;  % radious of bulls
equal.bullsize        =   cfg.visual.bullsize;


%% CREATE SOUND SAMPLES

% set waveform values
soundmatrix           = zeros(length(equal.freq0), ...                      % precreate sound matrix
                              length(equal.intloudness), ...
                              equal.sampdur * equal.samp_rate);
modwave               = createwaveform(equal.amplmod, ...                   % create waveform for amplitude mod
                                       equal.sampdur, ...
                                       equal.samp_rate); 
modwave               = ((equal.mod_index  * modwave) + 1)/ ...             % adjust modulatotion waveform (if needed)
                          (2/(equal.mod_index-equal.mod_min )) + equal.mod_min ;  
freq0waves            = equal.ampl * createwaveform(equal.freq0, ...        % create waveforms for all f0s
                                                    equal.sampdur, ...
                                                    equal.samp_rate);

% apply loudness modulation and set for all loudnesses
moddedwaves           = freq0waves .* modwave;
for iloud = 1:size(soundmatrix,2)
    soundmatrix(:,iloud,:)      =   reshape(moddedwaves * equal.intloudness(iloud), ... % set loudnesses and fit into soundmatrix
                                            length(equal.freq0), 1,[]);
end

%% DO ACTUAL EQUALISATION

% prepair everything
others                      = setdiff(1:size(soundmatrix,1),equal.refstim);     % all counds except for reference stim
if ~exist([pwd '/loudness/reff-loudness.mat'],'file')
    all_loudness            = zeros(1,size(soundmatrix,1));                     % set loudness values
    equal.ind_louds         = ones(1, size(all_loudness, 2)) * equal.refloud;   % set initial value to refloudness
else
    load (fullfile( pwd, 'loudness', ['reff-loudness.mat']), 'all_loudness');   % or load to already have a good guess
    equal.ind_louds         = all_loudness;
end
all_loudness(equal.refstim) = equal.refloud;                                    % set refference freq to refference loudness

% compare stimuli to ref
for i=1:length(others)

    % set start comparison value
    equal.ind_loud = equal.ind_louds(others(i));

    % compare loudness to refference and save
    if setup == 3
        [trainloudness,cc]      = compareloudness_MEG(squeeze(soundmatrix(equal.refstim,equal.refloud,:)), ...
                                                      squeeze(soundmatrix(others(i),:,:)), equal, w, screenrect);
    else
        [trainloudness,cc]      = compareloudness_MRI(squeeze(soundmatrix(equal.refstim,equal.refloud,:)), ...
                                                      squeeze(soundmatrix(others(i),:,:)), equal, w, screenrect);
    end
    all_loudness(others(i))  = trainloudness;

    if strcmp(cc,'ESCAPE')
            break;
    end 
end


%% SHUT DOWN AND SAVE DATA
save (fullfile( pwd, 'loudness', ppnum, [ppnum '-loudness.mat']), 'all_loudness', 'equal');

% cleanup
ShowCursor;     ListenChar;     sca;    clearvars;


end