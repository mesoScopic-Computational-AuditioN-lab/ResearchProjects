clc; clear all; close all
load('ff_range.mat', 'ff'); % load data

subz = {'S01_SES1', 'S02_SES1', 'S03_SES1', 'S04_SES1', 'S05_SES1', 'S06_SES1', 'S07_SES1', 'S08_SES1', 'S09_SES1', 'S10_SES1'};

% Parameters
pth = 0.0; % F selection thres in terms of percentiles of the empirical distribution
tunsteps = 20; freqstep = 1; % distance between to contiguous freq in log scale
freqscaling = 'log2'; % freq scaling in log2 or ln (log) space

%%% transform the muarray and related sigma arrays within a logistic domain
%%% Ideally we want to fit linear gaussians on a logarithmic frequency axis
%%% as otherwise gaussians need to be asymetric.
%%% can use log2 or log, but make sure octgrid is transformed accordingly
%%% (2,^ log2 etc). log would have a slight asymatric nature in the
%%% gaussians, where log2 will be symatrical as long as we are in octave
%%% space
if strcmp(freqscaling, 'log')
    inv_transform = @(x) exp(x);
    log_transform = @(x) log(x);
elseif strcmp(freqscaling, 'log2')
    inv_transform = @(x) 2.^x;
    log_transform = @(x) log2(x);
else
    error('Invalid freqscale value. Use "log" or "log2".');
end

%%% define the fwm in frequency bins 25 - 100 is equal to 0.49 octaves to
%%% 2.03 octaves, we take these linspace based on binnes to and later
%%% translate ff to octives. This is done to increase the population bins
%%% in lower frequencies compared to highter frequencies.
fwhm = linspace(25, 180, tunsteps); fwhm = sort(fwhm, 'descend');
subsample      = 1:freqstep:length(ff);          % grid of frequency bins
% Display range of FWHM based on scaling dimensions
log2_diff = diff(log_transform(ff));
sigma = log2_diff(1) * fwhm;                               % calculate corresponding sigma values
sigma_gaussian = sqrt(8*log(2)) * log2_diff(1) * fwhm;     % and gaussian value
disp(['FWHM ranging from ', num2str(sigma(end)), ' (sigma: ', num2str(sigma_gaussian(end)), ') to ', num2str(sigma(1)), ' (sigma: ', num2str(sigma_gaussian(1)), ')'])

%%% note the next few lines describe the stimulus domain, these can be in
%%% ln or in log2, dependent on freqscaling parameter
muarray_bins   = log_transform(ff);                        % all frequencies                                  
muarray        = log_transform(ff(subsample));             % frequency of every bin
mustep         = diff(log_transform(ff(subsample)));       % difference between to frequency bins


%%%% now we want to define a grind of sigams that will allow us to
%%%% generate our gaussias as exp(-[f0 - f]/2*sigma^2)
%%%% to mke this simple we generate our grid already as 1/(2*sigma^2)
%%%% in addition beacsue our frequency axis has become logarithmic we have
%%%% to go to a logarithmic axis as well!!! here comes the log again (not log2)
sigmagrid      = log_transform(2)./(mustep(1).*fwhm/2).^2; % grid of tuning width, in sigma units


%%%% here perhaps there was a mistake the exponent was n ot in the right
%%%% place in the rogiinal code of Tin and you were getting non-sensical
%%%% numbers....when converted to octaves [I think this was not really a 
%%%% problem as this was n ot sued to generarte the gaussians]
octgrid        = log2(inv_transform((fwhm-1)*mustep(1)));          % same grid converted to octaves


% create gaussian pRF dict
%%%% Note that this function will generate the grid of W as well as a long
%%%% prfmu and octave grid. CM and CS are linear spacing -10/10 vallues to
%%%% be used within psychopy for plotting purposes
[W, prfMU, prfS, prfO, CM, CS] = get_gausssian_weigthsZ_MM_f(muarray_bins, muarray, sigmagrid, octgrid);
% imagesc(W); plot(prfS); plot(prfO); plot(prfS)  


kfold = 5; nperm = 1000; % cross validation parameters

% Loop over all subjects
for subj = 1:length(subz)
    fprintf('Processing subject: %s\n', subz{subj});
    % Load Fmap
    dataFile = fullfile('/media/jorvhar/Data8T/MRIData/PreProc/', subz{subj}, 'Betas', 'BetasSEP_gm-subcortical_denoise0_cv1.mat');
    load(dataFile, 'BetasSEPTrain', 'FMapTrain', 'InfoVTCTrain');
    
    selvox = FMapTrain > pth;  % SELLECT VOXELS ABOVE PTH
    % selvox = ones(size(FMapTrain)); % TEMP: SELECT ALL VOXELS
    fmap = FMapTrain(selvox);
    BetasSEPTrain = BetasSEPTrain(:, selvox); % select voxels
    [ntrials, nvox] = size(BetasSEPTrain); % problem size
    
    % get the null distribution of the correlation coefficient between a
    % gaussian random vector and the gaussian pRF dictionary
    H0 = get_penaltybasedonRanks_MM(eye(size(W, 1)), [], [], nperm, W); 
    cv = cvpartition(ntrials, 'kfold', kfold); % start cross val
    
    %% KFOLD TESTING OF REPEATED TRIALS - BOTH GRIDSEARCH AND PERMUTATIONS

    for itcv = 6:(kfold + 1)
        disp(['Cross Val ', num2str(itcv)])
        if (itcv == kfold + 1) % last cv is whole data
            tr = logical(ones(ntrials, 1)); te = logical(ones(ntrials, 1)); % last cross val is whole data set
        else
            tr = cv.training(itcv); te = cv.test(itcv); % training and test
        end
        Xtr = BetasSEPTrain(tr, :); clear BetasSEPTrain; % Xte = BetasSEPTrain(te, :); % training and test betas
        Ftr = W(tr, :); % Fte = W(te, :); % training and test pRF dictionary
        cXtrFtr = corr(Xtr, Ftr); % corr between training betas and training dict
        [corrmax, selseed] = max(cXtrFtr'); % grid search: select the seed with max corr in the grid
        [locRanks, Wprf, pvalW] = fitGaussianPrfbasedonRanks_MM(eye(size(Ftr, 1)), Xtr, sigmagrid, muarray, H0, Ftr, []);
        indlinRanks = sub2ind(size(cXtrFtr), [1:nvox]', locRanks');
    end
    
    %% SAVE PRF MAPPING

    map = zeros(size(FMapTrain, 2), 2);
    CM = round(CM);
    map(selvox, 1) = CM(selseed); % mean and 
    map(selvox, 2) = CS(selseed); % mean and 
    saveICAMap(fullfile('/media/jorvhar/Data8T/MRIData/PreProc/', subz{subj}), 'Betas', map, ...
        {['Subject ' subz{subj} ': prfMU'] ['Subject ' subz{subj} ': TW']}, 'prf_gridsearch', InfoVTCTrain);
    
    map = zeros(size(FMapTrain, 2), 2);
    map(selvox, 1) = CM(locRanks); % mean and 
    map(selvox, 2) = CS(locRanks); % mean and 
    saveICAMap(fullfile('/media/jorvhar/Data8T/MRIData/PreProc/', subz{subj}), 'Betas', map, ...
        {['Subject ' subz{subj} ': prfMU'] ['Subject ' subz{subj} ': TW']}, 'prf_permutations', InfoVTCTrain);

    %% SAVE RAW TUNING VALLUES  
    %%%% here it depends on what you want as output 
    %%%% prfMU is the center frequency of the gaussian but not in Hz but in
    %%%% a log scale if you want Hz, then you need to do exp(prfMU)
    %%%% prfO on the other hand is the fwhm in octaves....
    %%%%
    %%%% for what comes after in your analyses I think you need to recreate
    %%%% the gaussians on a logarithmic axis of frequency qand thus I would
    %%%% save prfMU and prfS which allow you to regreate the gaussian with
    %%%% the formula 
    %%%% muarray        = log(ff(subsample)); 
    %%%% D2      = (prfMU' -  muarray).^2;
    %%%% gaussian = exp(-prfS .*D2);
    %%%% I have changed prfS to be that exat number
    map = zeros(size(FMapTrain, 2), 2);
    map(selvox, 1) = prfMU(locRanks); % Mean in ln or log2 space
    prfMU_inv = inv_transform(prfMU); map(selvox, 2) = prfMU_inv(locRanks); % Mean in hz
    map(selvox, 3) = prfS(locRanks); % STD sigma
    map(selvox, 4) = prfO(locRanks); % octive spread
    saveICAMap(fullfile('/media/jorvhar/Data8T/MRIData/PreProc/', subz{subj}), 'Betas', map, {'prfMU' 'prfMU_hz' 'prfS' 'prfO'}, 'prf_permutations_for_s2', InfoVTCTrain);

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%                                                                                                                    %%%  
%%%  PLOTTING FUNCTIONS TE SEE THE DISTRIBUTION EVOLUTION                                                              %%%
%%%    - fit by both gridsearch and permutation testing                                                                %%%
%%%    - fit distribution over frequencies and tws                                                                     %%%
%%%       * i.e. the sellection of the grid                                                                            %%%
%%%    - fit behaviour per voxel                                                                                       %%%
%%%                                                                                                                    %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% PLOT BRAIN LEVEL DISTRIBUTIONS
figure('Color',[1,1,1]);
subplot(221); hist3([CM(selseed),CM(locRanks)],[100 100] );    xlabel('grid search');ylabel('perm')
title('Central Frequency')
subplot(222); hist3([prfO(selseed),prfO(locRanks)],[100 100] ); xlabel('grid search');ylabel('perm')
title('Tuning Width')
subplot(223); h1 = histogram(prfMU(selseed),100); hold all; h2 = histogram(prfMU(locRanks),100);
title('Central Frequency')
legend([h1, h2], {'gridsearch', 'perm'});
delta = 0.05*min(prfO);
subplot(224); h1 = histogram(prfO(selseed),100); hold all; h2 = histogram(prfO(locRanks)+delta,100);
title('Tuning Width')
legend([h1, h2], {'gridsearch', 'perm'});

%% PLOT VOXEL LEVEL RESULTS
indlin  = sub2ind(size(cXtrFtr),[1:nvox]' , selseed' ); % put the index of the selected seed in linear form
corrhat = cXtrFtr(indlin); % extract the predictions for the each seed

locvox = find(corrhat>0.40  & prfO(selseed) >  3);
myvox = locvox(1)
sBetaseries = smooth( Xtr(:,myvox)./max(Xtr(:,myvox))  ) ;
[cord ord ] = sort(cXtrFtr(myvox,:),'ascend');
figure,
for it =1;
    subplot(221);
    plot(Xtr(:,myvox)./max(Xtr(:,myvox)),'LineWidth',2); hold all;
    hold all;
    l1 = plot( Ftr(:,selseed(myvox)),'-','LineWidth',2 ); title( corr(Xtr(:,myvox) ,  Ftr(:,ord(it)))); 
    l2 = plot( Ftr(:,locRanks(myvox)),'-','LineWidth',2 );hold off; title( corr(Xtr(:,myvox) ,  Ftr(:,ord(it)))); 

    disp(['prfMU: ' num2str(prfMU(selseed(myvox)))]);
    disp(['prf0: ' num2str(prfO(selseed(myvox)))]);

    legend([l1 l2],'gridsearch','perm')
    subplot(222);
    imagesc(octgrid,muarray,reshape(cXtrFtr(myvox,:),length(muarray),length(sigmagrid )));colorbar
    xlabel('tunning'); ylabel('Freq'); title('Cost Function Grid Search');
    hold all;plot(prfO(selseed(myvox)), prfMU(selseed(myvox)),'.r','MarkerSize',20)
    subplot(223);
    imagesc(octgrid,muarray,reshape(-log10(pvalW(myvox,:)),length(muarray),length(sigmagrid )));colorbar
    hold all;plot(prfO(locRanks(myvox)), prfMU(locRanks(myvox)),'.r','MarkerSize',20)
    xlabel('tunning'); ylabel('Freq'); title('Cost Function Permutations');
end