%% visualisation of single 

clear all;
clc;

% Fieldtip startup
addpath(genpath('/project/3018063.01/scripts_preproc/functions/'));
startup;

fiff_file = 'FIR_forward_adapted_activation.fif';
cfg = [];
cfg.dataset = fiff_file;
cfg.channel = 'MEG';
data1 = ft_preprocessing(cfg);

% set timelock
avg1 = ft_timelockanalysis([], data1);




% %% old shcool plotting
% 
% cfg = [];
% cfg.showlabels = 'yes';
% cfg.fontsize = 6;
% cfg.layout = 'CTF275_helmet.mat';
% % cfg.ylim = [-3e-13 3e-13];
% ft_multiplotER(cfg, avg1);
% 
% 
% %% fancy plotting
% 
% % full avarage
% cfg = [];
% cfg.colorbar = 'yes';
% cfg.layout = 'CTF275_helmet.mat';
% ft_topoplotER(cfg, avg1);
% 
% % at times
% cfg        = [];
% cfg.xlim   = avg1.time;  % Define 12 time intervals
% cfg.layout = 'CTF275_helmet.mat';
% ft_topoplotER(cfg, avg1);


% load the data
load('/project/3018063.01/preproc/sub-010/preproc/main/preproc-data-comp-cleaned-60hz.mat');
avg1.grad = data.grad;


%% cacluating the planar
% cfg                 = [];
% cfg.layout          = 'CTF275_helmet.mat';
% cfg.feedback        = 'yes';
% cfg.method          = 'template';
% cfg.neighbours      = ft_prepare_neighbours(cfg, avg1);

cfg                 = [];
cfg.method          = 'template';
cfg.template        = 'CTF275_neighb.mat';
% neighbours          = ft_prepare_neighbours(cfg, alldatacorrect);
neighbours          = ft_prepare_neighbours(cfg, avg1);

cfg                 = [];
cfg.method          = 'sincos';
cfg.neighbours      = neighbours;
data_planar         = ft_megplanar(cfg, avg1);

% compute the manitude planar gradient by combining horizontal and vertical
% comps
cfg = [];
avg_planar = ft_combineplanar(cfg, data_planar);


% plot differences
cfg = [];
cfg.zlim = 'maxmin';
cfg.colorbar = 'yes';
cfg.layout = 'CTF275_helmet.mat';
cfg.figure  = subplot(121);
ft_topoplotER(cfg, avg1)

colorbar; % you can also try out cfg.colorbar = 'south'

cfg.zlim = 'maxabs';
cfg.layout = 'CTF275_helmet.mat';
cfg.figure  = subplot(122);
ft_topoplotER(cfg, avg_planar);


% at times
cfg        = [];
cfg.xlim   = avg_planar.time;  % Define 12 time intervals
cfg.layout = 'CTF275_helmet.mat';
cfg.colormap = '*RdBu';
ft_topoplotER(cfg, avg_planar);



cfg        = [];
cfg.xlim   = avg_planar.time;  % Define 12 time intervals
cfg.xlim   = -0.05:0.05:.23333;
cfg.gridscale = 300;
cfg.style = 'straight';
cfg.layout = 'CTF275_helmet.mat';
cfg.colormap = '*RdBu';
cfg.marker =  'no';
ft_topoplotTFR(cfg, avg_planar);




