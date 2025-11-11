%% loopig version of beta estimation
clear variables; close all;

addpath('/home/jorvhar/Documents/MATLAB/NeuroElf_v10_5153')
addpath('/home/jorvhar/Desktop/fMRIEncoding')
addpath('/home/jorvhar/Desktop/fMRIEncoding/GLMdenoise-1.1/utilities')

% add pp9 after getting run 3 prt data ready
subz = {'S01_SES1', 'S02_SES1', 'S03_SES1', 'S04_SES1', 'S05_SES1', 'S06_SES1', 'S07_SES1', 'S08_SES1', 'S10_SES1'};
subz = {'S09_SES1'};
parsubj = '/media/jorvhar/Data8T/MRIData/PreProc/';


for sub = subz

    tic
    
    %%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%% ESTIMATE BETAS %%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%
    CrossValMat = [1 1 1 1 1 1 2]';
    %CrossValMat = [1 1 2 1 1 1]';    % for s9, missing run 3 - inserted dummy copy
    
    optstruct.Mask = 'gm-subcortical.msk';
    optstruct.Denoise = 0; % 1?
    
    optstruct.hrfthresh = 20;
    optstruct.thr_R2 = -0.1;
    
    optstruct.DenoisePredType = 'onoff-tonotopy';
    
    optstruct.noFIR = 1;
    optstruct.pttp = 4;
    optsruct.ons = 0;
    optstruct.nttp = 16;
    optstruct.pnr = 6;
    optstruct.pdsp = 1;
    optstruct.ndsp = 1;
    
    
    optstruct.FIRPredType = 'onoff-tonotopy';
    optstruct.ContrastFIR = [3 4];
    optstruct.CrossValFirFlag = 0;
    
    optstruct.SEPPredType = 'tonotopy';
    
    %optstruct.parallelOFF = 1;
    
    


    dirsubj = char(fullfile(parsubj, sub, '/'));
    getBetas(dirsubj,CrossValMat,optstruct);

    toc

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% 
% clear variables; close all;
% 
% addpath('/home/jorvhar/Documents/MATLAB/NeuroElf_v10_5153')
% addpath('/home/jorvhar/Desktop/fMRIEncoding')
% addpath('/home/jorvhar/Desktop/fMRIEncoding/GLMdenoise-1.1/utilities')
% 
% dirsubj = '/media/jorvhar/Data1/MRIData/PreProc/S02_SES1/';
% %dirsubj = '/media/jorvhar/Data1/MRIData/PreProc/S01_SES1/';
% 
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%% ESTIMATE BETAS %%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%
% CrossValMat = [1 1 1 1 1 1 2]';
% 
% optstruct.Mask = 'brainmask.msk';
% optstruct.Denoise = 0; % 1?
% 
% optstruct.hrfthresh = 20;
% optstruct.thr_R2 = -0.1;
% 
% optstruct.DenoisePredType = 'onoff-tonotopy';
% 
% optstruct.noFIR = 1;
% optstruct.pttp = 4;
% optsruct.ons = 0;
% optstruct.nttp = 16;
% optstruct.pnr = 6;
% optstruct.pdsp = 1;
% optstruct.ndsp = 1;
% 
% 
% optstruct.FIRPredType = 'onoff-tonotopy';
% optstruct.ContrastFIR = [3 4];
% optstruct.CrossValFirFlag = 0;
% 
% optstruct.SEPPredType = 'tonotopy';
% 
% optstruct.parallelOFF = 1;
% 
% getBetas(dirsubj,CrossValMat,optstruct);