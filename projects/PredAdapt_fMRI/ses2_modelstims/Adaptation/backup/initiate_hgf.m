function [hgf_path, add] = initiate_hgf()

hgf_path = 'C:\Users\jorie\OneDrive\Documenten\downloads\HGF\HGF';
addpath(hgf_path);

u = load('example_binary_input.txt');
%% 
% The inputs are simply a time series of 320 0s and 1s. This is the input sequence 
% used in the task of Iglesias et al. (2013), _Neuron_, *80*(2), 519-530.

scrsz = get(0,'ScreenSize');
outerpos = [0.2*scrsz(3),0.7*scrsz(4),0.8*scrsz(3),0.3*scrsz(4)];
figure('OuterPosition', outerpos)
plot(u, '.', 'Color', [0 0.6 0], 'MarkerSize', 11)
xlabel('Trial number')
ylabel('u')
axis([1, 320, -0.1, 1.1])


bopars = tapas_fitModel([],...
                         u,...
                         'tapas_hgf_binary_config',...
                         'tapas_bayes_optimal_binary_config',...
                         'tapas_quasinewton_optim_config');

sim = tapas_simModel(u,...
                     'tapas_hgf_binary',...
                     [NaN 0 1 NaN 1 1 NaN 0 0 1 1 NaN -2.5 -6],...
                     'tapas_unitsq_sgm',...
                     5,...
                     123456789);


tapas_hgf_binary_plotTraj(sim)

return;