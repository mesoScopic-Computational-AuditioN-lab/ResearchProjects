function JJ = get_penaltybasedonRanks_MM(FlowTR,~,~,nperm,W)
%% Input
% FlowTR    models sounds x features training data
% sarray    a vector of 1./sigma values for creating the gaussian dictionary
% freqArray a vector of the mu used for the gaussian dictionary 
% nperm     number of permutations

%% Output
% JJ        matrix of number of perm x number of gaussians, with gaussians ranked by their correlation for each permutation
%%
[nTR,nrcomp] =  size(FlowTR);
% freqBins = length(freqArray);
% [D,~,~] = get_gausssian_weigthsZ_MM(linspace(1,nrcomp,nrcomp),freqArray,sarray);% creates a dictionary of gaussian functions
H = zscore(FlowTR*W);                           %project training data on gaussian functions
%%compute correlation for permutations
JJ = zeros(nperm,size(W,2));      %initialize matrix to store output permutations
X  = zscore(randn(nTR,nperm));         %generate random data
cc = (X'*H)./(nTR-1);                  %cross product is identical to correlation; cc = correlation all gaussians - all permutations
[~, jj] = sort(cc','descend'); jj = jj';        %sort correlation per permutation (order gaussians)
%%assign rank
pp = 1:size(cc,2);                              %vector of 1:number of gaussians
for itperm = 1:nperm
    JJ(itperm,jj(itperm,:)) = pp;               %per permutation, assign rank to gaussians: which ranked first, which last?
end
