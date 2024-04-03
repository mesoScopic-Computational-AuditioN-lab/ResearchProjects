%% visualisation of single 

clear all;
clc;

% Fieldtip startup
addpath(genpath('/project/3018063.01/scripts_preproc/functions/'));
startup;

% subject idds
sub_id     = 'sub-001';

% get all the evoked
folder = '/project/3018063.01/preproc';
files = dir([folder '/' sub_id '/evoked']);
fileNames = {'FIR_forward_adapted_activation', 'FIR_onoff', ...
             'FIR_pred_prob', 'FIR_surprisal', 'MNE_forward_adapted_activation', 'MNE_FIR_onoff', ...
             'MNE_pred_prob', 'MNE_surprisal'};
out_prefix = 'grad_';

% take grad information
grad_folder = sprintf('/project/3018063.01/raw/%s/ses-meg01/meg/', sub_id);
grad_fn = dir([grad_folder '*.ds']);

% loop over filenames
for fn = 1: numel(fileNames)

    % load current evoked
    fiff_file = [fileNames{fn} '-' sub_id '.fif'];
    fiff_path = [folder '/' sub_id '/evoked/' fiff_file];
    cfg = [];
    cfg.dataset = fiff_path;
    cfg.channel = 'MEG';
    data1 = ft_preprocessing(cfg);

    % load grad information for localization
    grad = ft_read_sens([grad_folder grad_fn.name], 'senstype', 'meg');
    data1.grad = grad;
    
    % prepair megplanar
    cfg                 = [];
    neighbours          = load('ctf275_neighb.mat');
    cfg.neighbours      = neighbours.neighbours;
    cfg.feedback        = 'yes';
    cfg.method          = 'sincos';
    avgplanar           = ft_megplanar(cfg, data1);

    % combine planar into grad magnitude
    cfg = [];
    avgplanarComb = ft_combineplanar(cfg, avgplanar);
    data = ft_timelockanalysis([], avgplanarComb);
    % data = avgplanarComb;

    %% Plotting if wanted set timelock
    cfg        = [];
    cfg.xlim   = data.time;  % Define 12 time intervals
    cfg.xlim   = -0.05:0.05:.23333;
    cfg.gridscale = 300;
    cfg.style = 'straight';
    cfg.layout = 'CTF275_helmet.mat';
    cfg.colormap = '*RdBu';
    cfg.marker =  'no';
    ft_topoplotTFR(cfg, data);
    title(strrep(fileNames{fn}, '_', ' '), 'FontSize', 14, 'FontWeight', 'bold');

    %% save as fif
    cfg = [];
    fiff_output = [out_prefix fileNames{fn} '-' sub_id '.mat'];
    fiff_output_path = [folder '/' sub_id '/evoked/' fiff_output ];
    % fieldtrip2fiff(fiff_output_path, avg_planar);
    % save(fiff_output_path,"data")  % uncheck for saving

end