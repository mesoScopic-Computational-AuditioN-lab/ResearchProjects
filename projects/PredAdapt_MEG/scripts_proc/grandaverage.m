%% visualisation of single 

clear all;
clc;

% Fieldtip startup
addpath(genpath('/project/3018063.01/scripts_preproc/functions/'));
startup;

% subids
sub_ids     = {'sub-001', 'sub-002','sub-004','sub-005', ...
               'sub-007', 'sub-008', 'sub-009', 'sub-010', 'sub-011', ...
               'sub-012', 'sub-013', 'sub-014'};

% Conditions
conditions = {'FIR_forward_adapted_activation', 'FIR_onoff', ...
              'FIR_pred_prob', 'FIR_surprisal', 'MNE_forward_adapted_activation', 'MNE_FIR_onoff', ...
              'MNE_pred_prob', 'MNE_surprisal'};
conditions = {'MNE_surprisal'};

% Folder containing the data
folder = '/project/3018063.01/preproc';
out_prefix = 'grad_';

for j = 1:length(conditions)

    % Initialize a cell array to hold all data structures
    allData = {};

    condition = conditions{j};

    % Loop over each subject and condition
    for i = 1:length(sub_ids)
        sub_id = sub_ids{i};

        file_name = [out_prefix condition '-' sub_id '.mat'];  % Construct the file name
        file_path = fullfile(folder, sub_id, 'evoked', file_name);  % Construct the full file path

        % Check if the file exists
        if exist(file_path, 'file')
            % Load the data
            loadedData = load(file_path, 'data');
            allData{end+1} = loadedData.data;  % Append the data to the cell array
        else
            warning(['File not found: ' file_path]);
        end

    end
        
    % plot condition grand average
    cfg = [];
    cfg.keepindividual = 'no';  % Change to 'yes' if you want to keep individual data
    grandAverage = ft_timelockgrandaverage(cfg, allData{:});
    
    %% Plotting if wanted set timelock
    cfg        = [];
    cfg.xlim   = grandAverage.time;  % Define 12 time intervals
    cfg.xlim   = -0.05:0.05:.23333;
    % cfg.zlim   = 'maxabs';
    % cfg.zlim   = [min(grandAverage.avg, [], 'all') max(grandAverage.avg, [], 'all')];
    cfg.zlim   = [0.1 1.3];
    % cfg.baseline = [-0.02 0.01];
    % cfg.baselinetype = 'relative';
    cfg.gridscale = 300;
    cfg.style = 'straight';
    cfg.layout = 'CTF275_helmet.mat';
    cfg.colormap = '*RdBu';
    % cfg.colormap = 'coolwarm';
    cfg.marker =  'no';
    ft_topoplotTFR(cfg, grandAverage);
    title(strrep(condition, '_', ' '), 'FontSize', 14, 'FontWeight', 'bold');

end



% Now you can plot the grand average