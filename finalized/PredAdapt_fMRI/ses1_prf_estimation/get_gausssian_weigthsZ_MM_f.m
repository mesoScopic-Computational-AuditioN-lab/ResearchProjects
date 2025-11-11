function [W MU S O CM CS]  = get_gausssian_weigthsZ_MM(f,muarray,sarray,octavearray)

musteps     = numel(muarray);
ssteps      = numel(sarray);
colormapMU  = sort(linspace(-10 , 10, musteps),'descend');
colormapS  =  sort(linspace(-10 , 10, ssteps),'descend');
%muarray  = linspace(min(f),max(f),musteps);
ns       = length(sarray);
sarray1  = 1:ns;
W       = [];
MU      = [];
CM      = [];
S       = []; O = []; CS = [];
D2      = (f' -  muarray).^2;
W       = zeros(length(f),musteps*length(sarray));
for it = 1:ns

    %%% our sarray is already created to be in the frequency axis but with
    %%% a log spacing (NOT OCTAVES) this is the same smacing of f-muarray 
    %%% so sarray is correct!
    %%%% I also chenged it so that you save the actual sarray that makes it
    %%%% simpler for you to recreate the gaussians
   
    W(:,((it-1)*musteps+1):(it*musteps))   = exp(-sarray(it).*D2);
    S   = [S ; 0.*muarray' + sarray(it)];
    MU  = [MU;  muarray'];
    CM  = [CM; colormapMU'  ];
    O   = [O ; 0.*muarray' + octavearray(it)];
    CS  = [CS; 0.*muarray' + colormapS(it)];
end