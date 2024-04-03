function [Mnew,nblocks]=counterbalance(M,tpc,B,subdivs,X,methods,interleave)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Provides a matrix of counterbalanced conditions (on different rows)
% for multiple trials (in different columns).
%
% Example: conditionmatrix=counterbalance([2 3],2,[3 4],2,[7 9],'full');
%
%
% INPUT:
%
% - M:   Main conditions (array with number of levels per factor).
%
% - tpc: Number of trials per condition that are randomly intermixed
%
% - B: Secondary conditions for pseudo random designs. These are fully
%      counterbalanced with the main conditions but randomly assigned
%      to a subdivision (e.g., counterbalanced over entire experiment 
%      (or subdivision) but not within experimental blocks). 
%
% - subdivs: number of fully counterbalanced subsequent subdivisions.
%      This includes the conditions provided in [B]. If B is left empty 
%      the number of subdivisions is equal to the number of blocks.
%
% - X: Uncounterbalanced conditions. Optimized for equal prevalence per
%      condition, as far as is allowed for by the number of levels in the
%      factor, and the total number of trials as determined by the other
%      input variables.
%
% - methods: Choose amongst the following restriction methods for the
%            uncounterbalanced condition(s) X. If you provide one method
%            this will apply to all values within X. You can also provide
%            a cell array with different methods for each instance of X.
%
%             'block':    same values of X in all blocks
%             'subdiv:    same values of X in all subdivisions
%             'full':     X is randomly distributed across whole matrix
%
% - interleave: Interleave subdivisions such that [A1 A2 B1 B2] becomes
%               [A1 B1 A2 B2] or [A1 B2 A2 B1], where A and B are identical
%               subdivisions.
%
%
%   Jorie van Haren, adopted from Surya Gayet
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if ~exist('subdivs','var')|subdivs<1; subdivs=1; end %#ok<*OR2>
if ~exist('tpc','var')|tpc<1; tpc=1; end
if ~exist('B','var')|max(B)<1; B=1; end
if ~exist('interleave','var'); interleave=0; end
if ~exist('X','var'); X=[]; end
if exist('methods','var')&iscell(methods)==0; methods={methods}; end %#ok<AND2>

% Create balanced design
Mnew=zeros(sum([M>1 B>1]),prod([M tpc B subdivs]));
for k=1:subdivs
    Mbal=cbsimple([1 M],tpc,1); 

    Mbal(1,:)=1:size(Mbal,2);
    if sum(B>1)>0;
        Msec=cbsimple(B,1,0);
        Mexp=[Expand(Mbal,size(Msec,2),1); repmat(Msec,1,prod([M tpc]))];
        sameperblock=sum(Mexp(1,:)==1);
        Mbal=zeros(size(Mexp));
        Mexp=Mexp(:,randperm(size(Mexp,2)));
        loca=1;
        for j=1:sameperblock;
            findis=Shuffle(1:max(Mexp(1,:)));
            for i=1:max(Mexp(1,:))
                findi=findis(i);
                fincol=find(Mexp(1,:)==findi,1,'first');
                Mbal(1:size(Mexp,1),loca)=Mexp(:,fincol);
                Mexp(1,fincol)=0;
                loca=loca+1;
            end
        end
    end
    Mnew(:,(k-1)*size(Mbal,2)+1:size(Mbal,2)*k)=Mbal(2:end,:);
end

% compute number of blocks
nblocks=size(Mnew,2)/prod([M tpc]); ntrials=size(Mnew,2);

% add uncounterbalanced factors
if sum(X>1)>0;
    if ~exist('methods','var'); methods={'full'}; end
    if length(methods)==1;methods=repmat(methods,[1 length(X)]);end
    Mnew=[Mnew; zeros(size(X,2),size(Mnew,2))];
    for i=1:length(X)
        method=methods{i};
        switch method
            case 'block'
                trilim=prod([tpc M]);
            case 'subdiv'
                trilim=size(Mnew,2)/subdivs;
            case 'full'
                trilim=size(Mnew,2);
        end
        for j=1:ntrials/trilim;
            startp=(j-1)*trilim+1; endp=trilim*j;
            Xitemp=Shuffle(Expand(1:X(i),ceil(trilim/X(i)),1));
            Mnew(end-(abs(i-(length(X)))),startp:endp)=Xitemp(1,1:trilim);
        end
    end
end

if interleave>0; Mi=zeros(size(Mnew,1),size(Mnew,2)/(nblocks/subdivs),nblocks/subdivs);
    for nblock=1:nblocks
        startp=(nblock-1)*(ntrials/nblocks)+1; endp=startp+(ntrials/nblocks)-1;
        whichiti=mod(nblock-1,(nblocks/subdivs))+1; whichsubp=ceil(nblock/(nblocks/subdivs)); 
        startbl=(whichsubp-1)*(ntrials/nblocks)+1; endbl=startbl+(ntrials/nblocks)-1;
        Mi(:,startbl:endbl,whichiti)=Mnew(:,startp:endp);
    end
    Mnew=reshape(Mi,size(Mnew,1),size(Mnew,2));
end


function mbal=cbsimple(M,tpc,rando)
if nargin<3; rando=1; end
if nargin<2; tpc=1; end
M=[M 1]; mbal=ones(size(M-1,2),prod(M));
for i = size(mbal,1):-1:1
    mbal(i,:)=repmat(Expand((1:M(i)),prod(M(1:i-1)),1),1,prod(M(i+1:end)));
end
mbal(end,:)=[]; mbal=repmat(mbal,1,tpc);
if rando
   mbal=mbal(:,randperm(size(mbal,2))); 
end