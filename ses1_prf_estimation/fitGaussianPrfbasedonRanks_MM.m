% function [W,BestFreq,BestStd,cTR,D] = fitGaussianPrfbasedonRanks_MM(FlowTR,bLowTR,sarray,farray,H0,W)
function [locmax , W , pval] = fitGaussianPrfbasedonRanks_MM(FlowTR,bLowTR,~,~,H0,W,opt)

%% open parallel pool
% mv = version('-date');
% if str2double(mv(end-1:end)) >=14
%     if isempty(gcp('nocreate'))
%         %parpool
%         parpool('local', 12);
%     end
% else
%     if parpool('size')==0
%         parpool
%     end
% end
% cTRout = opt.cTR;
%% get gaussian dictionary
[nTR, nrcomp] = size(FlowTR);
% [W,MU,S]= get_gausssian_weigthsZ_MM(linspace(1,nrcomp,nrcomp),farray,sarray);% creates a dictionary of gaussian functions
% freqBins = length(farray);
% if numel(farray) == 1
%     MU = ones(size(MU));
% end

%% project data on gaussians
nvox    = size(bLowTR,2);
xW0     = FlowTR*W;                       % Fit in training
alpha   = sqrt(nTR-1)./sqrt(sum(xW0.^2)); % normalization factor for making std of pred eq to 1
W       = alpha.*W;                       % scaling the gaussian Prf by a factor that makes the prediction to have var = 1
BhatG   = zscore(FlowTR*W);                       % recalculating the fit with the normalized Prf
cTR     = (bLowTR'*BhatG)./(nTR-1);       % this is the correlation coeff size: nvox,ngaussians
clear bLowTR;
%% per voxel, compute rank of each gaussian for fit on data
nbatch = floor(size(cTR,1)/10000); % if 0, you will get an error, so please check
jj = [];
for i=1:nbatch
    if i<nbatch
        [~,temp] = sort(cTR((i-1)*nbatch+1:i*nbatch,:)','descend'); 
    else
        [~, temp] = sort(cTR((i-1)*nbatch+1:size(cTR,1),:)','descend'); 
    end
    jj=[jj;temp'];
end

% maxcorr = max(cTR');
JJ      = zeros(nvox,size(cTR,2));
pp = 1:size(cTR,2);
clear cTR temp;
% vector of 1:number of gaussians
for itperm = 1:nvox
    JJ(itperm,jj(itperm,:)) = pp;
end

%% computing the pvalues
pval = zeros(size(JJ)); %initialize matrix
nperm = size(H0,1);
for ath = 1:size(H0,2) %loop through gaussians: SLOW!
   % pval{:,ath}    = ( sum(( JJ(:,ath)' >= H0(:,ath)  ))+1   )./(nperm+1); %count how often the rank of the data is larger than H0 (rank expected by chance based on perm)
    pval(:,ath)    = ( sum(( JJ(:,ath)'  >= H0(:,ath)  ))+1   )./(nperm+1); %count how often the rank of the data is larger than H0 (rank expected by chance based on perm)
end
%pval = cell2mat(pval);

% %% adjust to account for multiple locations having min pval
  [~, locmax] = min(pval');
% for ivox = 1:numel(locmax)
%     indthis = find(pval(ivox,:) == pval(ivox,locmax(ivox)));
%     
%     if numel(indthis) > 1
%         [indMU,indSIG] = ind2sub([freqBins,size(pval,2)/freqBins],indthis);
%         meanMU = round(mean(indMU));
%         meanSIG = round(mean(indSIG));
%         locmax(ivox) = sub2ind([freqBins,size(pval,2)/freqBins],meanMU,meanSIG);
%     end
% end

%% prepare output
W           = W(:,locmax); % indexing the gaussian dictionary with the index of the best gaussian for each voxel
% BestFreq    = MU(locmax);
% BestStd     = S(locmax);

%% close parallel pool
% mv = version('-date');
% if str2double(mv(end-1:end)) >=14
%     poolobj = gcp('nocreate');
%     if ~isempty(poolobj)
%         delete(poolobj);
%     end
% else
%     if parpool('size')~=0
%         parpool CLOSE
%     end
% end

