clear all;
clc;

% Fieldtip startup
addpath(genpath('/project/3018063.01/scripts_preproc/functions/'));
startup;

% Define your file names
fif_files = {'FIR_forward_adapted_activation.fif', 'FIR_surprisal.fif', 'FIR_pred_prob.fif'};

% Initialize a cell array to hold the preprocessed data from each file
preprocessed_data = cell(1, length(fif_files));

% Loop through each file
for iFile = 1:length(fif_files)
    % Specify the dataset
    cfg = [];
    cfg.dataset = fif_files{iFile};
    cfg.trl = [1, inf, 0]; % Define the whole file as a single trial
    cfg.channel = 'all'; % Use all channels, modify if needed

    % Preprocess the data
    preprocessed_data{iFile} = ft_preprocessing(cfg);
end

% Now, append all the data into one structure assuming they can be combined directly
cfg_append = [];
combined_data = ft_appenddata(cfg_append, preprocessed_data{:});

%%



cfg = [];
cfg.showlabels = 'no';
cfg.fontsize = 6;
cfg.layout = 'CTF151_helmet.mat';
cfg.baseline = [-0.2 0];
cfg.xlim = [-0.2 1.0];
cfg.ylim = [-3e-13 3e-13];
ft_multiplotER(cfg, avgFC, avgIC, avgFIC);





% Define your file names
fif_files = {'FIR_forward_adapted_activation.fif', 'FIR_surprisal.fif', 'FIR_pred_prob.fif'};

% Initialize a cell array to hold the preprocessed data from each file
preprocessed_data = cell(1, length(fif_files));

% Initialize a struct to hold the averaged data
averaged_data = struct();

% Loop through each file
for iFile = 1:length(fif_files)
    % Specify the dataset
    cfg = [];
    cfg.dataset = fif_files{iFile};
    cfg.trl = [1, inf, 0]; % Define the whole file as a single trial
    cfg.channel = 'all'; % Use all channels, modify if needed

    % Preprocess the data
    preprocessed_data{iFile} = ft_preprocessing(cfg);

    % Perform timelock analysis to get the average
    cfg_timelock = [];
    averaged_data.(sprintf('trial%d', iFile)) = ft_timelockanalysis(cfg_timelock, preprocessed_data{iFile});
end

% Now, append all the data into one structure assuming they can be combined directly
cfg_append = [];
combined_data = ft_appenddata(cfg_append, preprocessed_data{:});

%% Old school plotting

cfg = [];
cfg.showlabels = 'no';
cfg.fontsize = 6;
cfg.layout = 'CTF275_helmet.mat';
% Convert the fields of averaged_data to a cell array
avg_data_cell = struct2cell(averaged_data);
% Call ft_multiplotER using the data in avg_data_cell
ft_multiplotER(cfg, avg_data_cell{:});


%% topo map

