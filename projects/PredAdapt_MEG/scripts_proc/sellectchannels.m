%% visualisation of single 

clear all;
clc;

% Fieldtip startup
addpath(genpath('/project/3018063.01/scripts_preproc/functions/'));
startup;


% Load the layout
cfg = [];
cfg.layout = 'CTF275_helmet.mat'; % specify your layout file
layout = ft_prepare_layout(cfg);

% Modify cfg to not show labels
cfg.box = 'no';   %      = string, 'yes' or 'no' whether box should be plotted around electrode (default = 'yes')
cfg.mask  = 'no'; %       = string, 'yes' or 'no' whether the mask should be plotted (default = 'yes')

% Plot the layout
ft_layoutplot(cfg, layout);

% Now you can visually inspect the layout and note down the sensors you are interested in.
