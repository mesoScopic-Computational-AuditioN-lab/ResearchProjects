clc;clear all;close all
load('/media/jorvhar/Data/MRIData/PreProc/S02_SES1/Betas/4Tin.mat'); % load data
load('/media/jorvhar/Data/MRIData/PreProc/S02_SES1/Betas/BetasSEP_AC_denoise1_cv1.mat','FMapTrain','InfoVTCTrain'); % load Fmap
pth           = 1.6; % F selection thres in terms of percentiles of the empirical distribution: this code also works for all vox
selvox        = FMapTrain  > pth;
fmap          = FMapTrain(selvox);
BetasSEPTrain = BetasSEPTrain(:,selvox); % select voxels
[ntrials nvox] = size(BetasSEPTrain);    % problem size
%% create gaussing dictionary
tunsteps       = 10; freqstep = 1;% distance between to contigous freq in log scale
fwhm           = linspace(25,(100), tunsteps);   fwhm = sort(fwhm,'descend');% grid of fwhm
subsample      = 1:freqstep:length(ff);          % grid of frequency bins
mustep         = diff(log(ff(subsample)));       % difference between to frequency bins
sigmagrid      = log(2)./(mustep(1).*fwhm/2).^2; % grid of tuning width, in sigma units
octgrid        = log2(exp(1))*fwhm*mustep(1);          % same grid converted to octaves
muarray_bins   = log(ff);                        % all frequencies
muarray        = log(ff(subsample));             % frequency of every bin

% create gaussian pRF dict
[W, prfMU, prfS, prfO, CM]      = get_gausssian_weigthsZ_MM(muarray_bins,muarray,sigmagrid,octgrid);%
% plot for checking: we used the central freq for this plot, here we show
% from the narrowest to the broadest
ii = find(prfMU == prfMU(length(muarray)/2));
figure('Color',[1,1,1]);plot(muarray_bins,W(:,ii ),'-','LineWidth',2); hold all; 
% % %%
% for it = 1:(length(octgrid))
%     loc     = abs(W(:,(it-1)*24 + 12 )-0.5) < 1e-4;
%     r       = ff(loc);
%     oct(it)   = log2( (r(2)/r(1)  ));    
% end
%% cross validation for pRF estimation
kfold         = 5;nperm         = 1000;
% get the null distribution of the correlation coefficient between a
% gaussian random vector and the gaussian pRF dictionary
H0            = get_penaltybasedonRanks_MM(eye(size(W,1)),[],[],nperm,W); 
cv = cvpartition(ntrials,'kfold',kfold);% start cross val
figure('Color',[1,1,1])
for itcv = 6:(kfold+1)
    disp(['Cross Val ', num2str(itcv) ])
    if(itcv == kfold+1 )  % last cv is whole data
        tr = logical(ones(ntrials,1)); te = logical(ones(ntrials,1));;% last cross val is whole data set
    else
        tr      = cv.training(itcv);   te = cv.test(itcv); % training and test
    end
    Xtr     = BetasSEPTrain(tr,:);clear BetasSEPTrain;% Xte = BetasSEPTrain(te,:);% training and test betas
    Ftr     = W(tr,:);            % Fte = W(te,:); % training and test pRF dictionary
    cXtrFtr = corr(Xtr,Ftr);     % corr between trainining betas and training dict
    [corrmax , selseed]      = max(cXtrFtr'); % grid search: select the seed with max corr in the grid
%     cXteFte = corr(Xte,Fte);     % corr in the test data
%     indlin  = sub2ind(size(cXtrFtr),[1:nvox]' , selseed' ); % put the index of the selected seed in linear form
%     corrhat = cXteFte(indlin); % extract the predictions for the each seed
    % based on ranks: do the same using the Ho for selcting the pRF
%     opt.cTR = cXtrFtr;
    [locRanks,  Wprf , pvalW]  = fitGaussianPrfbasedonRanks_MM(eye(size(Ftr,1)), Xtr  ,sigmagrid,muarray,H0,Ftr,[]);
    indlinRanks  = sub2ind(size(cXtrFtr),[1:nvox]' , locRanks' );
%     corrTrRank   =  cXtrFtr(indlinRanks);% corr in the training data of the seed selected with ranks
%     corrTeRank   =  cXteFte(indlinRanks);% corr in the test data for the seed in selected with rank 
%      d(itcv) = sum(corrTeRank > 0.2)  - sum(corrhat > 0.2);
%     d(itcv) = mean(corrTeRank)  - mean(corrhat );

    % plot for comparing both methods
%     subplot(2,6,itcv);histogram(corrhat,100); hold all;histogram(corrTeRank,100); 
%     plot([mean(corrhat) mean(corrhat)],ylim(),'LineWidth',2);title(d(itcv))
%     subplot(2,6,itcv+6);
%     plot(fmap, corrhat,'.' );hold all;   
%     plot(fmap, corrTeRank,'r.' );xlabel('f map'); ylabel('corr')    
end
%% explore with graph what is happening
% locvox = find(corrhat>0.2  & prfO(selseed) <  0.6 );
% myvox = locvox(1)
% sBetaseries = smooth( Xtr(:,myvox)./max(Xtr(:,myvox))  ) ;
% [cord ord ] = sort(cXtrFtr(myvox,:),'descend');
% figure,
% for it =1;
%     subplot(221);
%     plot(Xtr(:,myvox)./max(Xtr(:,myvox)),'LineWidth',2); hold all;
%     hold all;
%     l1 = plot( Ftr(:,selseed(myvox)),'-','LineWidth',2 ); title( corr(Xtr(:,myvox) ,  Ftr(:,ord(it)))); 
%     l2 = plot( Ftr(:,locRanks(myvox)),'-','LineWidth',2 );hold off; title( corr(Xtr(:,myvox) ,  Ftr(:,ord(it)))); 
%     legend([l1 l2],'gridsearch','perm')
%     subplot(222);
%     imagesc(octgrid,muarray,reshape(cXtrFtr(myvox,:),length(muarray),length(sigmagrid )));colorbar
%     xlabel('tunning'); ylabel('Freq'); title('Cost Function Grid Search');
%     hold all;plot(prfO(selseed(myvox)), prfMU(selseed(myvox)),'.r','MarkerSize',20)
%     subplot(223);
%     imagesc(octgrid,muarray,reshape(-log10(pvalW(myvox,:)),length(muarray),length(sigmagrid )));colorbar
%     hold all;plot(prfO(locRanks(myvox)), prfMU(locRanks(myvox)),'.r','MarkerSize',20)
%     xlabel('tunning'); ylabel('Freq'); title('Cost Function Permutations');
% 
% %     pause();
% end

%%
figure('Color',[1,1,1]);
subplot(221); hist3([CM(locRanks),CM(locRanks)],[100 100] );    xlabel('grid search');ylabel('perm')
title('Central Frequency')
subplot(222); hist3([prfO(selseed),prfO(locRanks)],[100 100] ); xlabel('grid search');ylabel('perm')
title('Tuning Width')
subplot(223); histogram(prfMU(selseed),100); hold all; histogram(prfMU(locRanks),100);
title('Central Frequency')
delta = 0.05*min(prfO);
subplot(224); histogram(prfO(selseed),100); hold all; histogram(prfO(locRanks)+delta,100)
title('Tuning Width')
%% 
map            = zeros(size(FMapTrain,2),2);
CM = round(CM);
map(selvox,1)  = CM(selseed) ;  % mean and 
map(selvox,2)  = prfO(selseed)  ;  % mean and 
saveICAMap('/media/jorvhar/Data/MRIData/PreProc/S02_SES1/','Betas',map ,'prf_gridsearch','prf_gridsearch',InfoVTCTrain);

map = zeros(size(FMapTrain,2),2);
map(selvox,1)  = CM(locRanks) ;  % mean and 
map(selvox,2)  = prfO(locRanks)  ;  % mean and 
saveICAMap('/media/jorvhar/Data/MRIData/PreProc/S02_SES1/','Betas',map ,'prf_permutations','prf_permutations',InfoVTCTrain);

