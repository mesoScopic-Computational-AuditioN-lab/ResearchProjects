addpath(genpath('/project/3018063.01/scripts_preproc/functions/'));
startup

dir_cur = pwd;
idx_str = strfind(dir_cur, 'Code');

cfg = [];
cfg.layout = 'CTF275_helmet.mat';
layout_meg = ft_prepare_layout(cfg);

figure
ft_plot_layout(layout_meg);