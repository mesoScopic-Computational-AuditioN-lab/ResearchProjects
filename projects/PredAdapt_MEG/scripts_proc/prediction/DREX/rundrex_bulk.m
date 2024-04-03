% Wrapper to loop over all participants in pps and execute rundrex_stims for all
% to use make sure xx_stimdf.mat exists and contains the stimuli / needed grids

clear all;
clc;

% pilot ids
sub_ids     = {'1', '2','4','5', ...
               '7', '8','9','10', '11', ...
               '12', '13', '14'};

for pp = 1: numel(sub_ids)
    % grab current sub id
    sub_id = sub_ids{pp};

    % run drex
    reqstring = 'mem=20gb,walltime=04:00:00,nodes=1';
    myqsub('rundrex_stims', reqstring, {pp})

    %rundrex_stims(sub_id, '/project/3018063.01/beh/data');
end