function [W MU S O CM]  = get_gausssian_weigthsZ_MM(f,muarray,sarray,octavearray)

musteps     = numel(muarray);
colormapMU  = sort(linspace(-10 , 10, musteps),'descend');
%muarray  = linspace(min(f),max(f),musteps);
ns       = length(sarray);
sarray1  = 1:ns;
W       = [];
MU      = [];
CM      = [];
S       = []; O = [];
D2      = (f' -  muarray).^2;
W       = zeros(length(f),musteps*length(sarray));
for it = 1:ns
W(:,((it-1)*musteps+1):(it*musteps))   = exp(-sarray(it).*D2);
S   = [S ; 0.*muarray' + 1./sqrt(2.* sarray1(it) )];
MU  = [MU;  muarray'];
CM  = [CM; colormapMU'  ];
O   = [O ; 0.*muarray' + octavearray(it)];
end

