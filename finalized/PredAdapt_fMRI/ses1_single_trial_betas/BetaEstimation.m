clear variables; close all;

addpath('/home/jorvhar/Documents/MATLAB/NeuroElf_v10_5153')
addpath('/home/jorvhar/Desktop/fMRIEncoding')
addpath('/home/jorvhar/Desktop/fMRIEncoding/GLMdenoise-1.1/utilities')

dirsubj = '/media/jorvhar/Data8T/MRIData/PreProc/S03_SES1/';


%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% ESTIMATE BETAS %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%
CrossValMat = [1 1 1 1 1 1 2]';

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

optstruct.parallelOFF = 1;

getBetas(dirsubj,CrossValMat,optstruct);


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