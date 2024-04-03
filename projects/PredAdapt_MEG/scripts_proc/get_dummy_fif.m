% cleanup
clear functions;

% Startup
addpath(genpath('/project/3018063.01/scripts_preproc/functions/'));
startup


% load data for dummy
load('/project/3018063.01/preproc/sub-010/preproc/main/preproc-data-comp-cleaned-60hz.mat')

% sellect only the first data
cfg = [];
cfg.trials = (data.trialinfo(:,1)==1);
data_t1 = ft_selectdata(cfg, data);

% convert to fif
fiff_file = '/project/3018063.01/preproc/sub-010/preproc/main/ctf-peproc-main-10.fif';
fieldtrip2fiff(fiff_file, data_t1);


%% ANUSED!!