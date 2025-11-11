function saveICAMap(dirsubj,subdir,map,mapname,fname,InfoVTC)

%%%% saves maps to file
%%%% map = nrvox x nrmaps

% pdsize = 3; % how many voxels to remove from all directions

ica = xff('vmp');

for m = 1:size(map,2)
    if m~=1
        ica.Map(m) = ica.Map(m-1);
    else
        ica.resolution = InfoVTC.Resolution;
        if isfield(InfoVTC,'BBox')
            ica.XStart = InfoVTC.BBox(1);
            ica.XEnd = InfoVTC.BBox(2);
            ica.YStart = InfoVTC.BBox(3);
            ica.YEnd = InfoVTC.BBox(4);
            ica.ZStart = InfoVTC.BBox(5);
            ica.ZEnd = InfoVTC.BBox(6);
        end
    end
    ica.Map(m).VMPData = zeros(InfoVTC.DimVTC);
    ica.Map(m).VMPData(InfoVTC.voxVTC)= map(:,m);

%     maskarray =  zeros(InfoVTC.DimVTC);
%     maskarray(pdsize+1:end-pdsize,pdsize+1:end-pdsize,pdsize+1:end-pdsize) = 1; % set boardes to zero
% 
%     ica.Map(m).VMPData = maskarray.*ica.Map(m).VMPData;


    if iscell(mapname)
        ica.Map(m).Name = mapname{m};
    else
        ica.Map(m).Name = [mapname,'_',num2str(m)];
    end
end
disp(['saving: ',fullfile(dirsubj,subdir,fname),'.vmp']) 
ica.SaveAS([fullfile(dirsubj,subdir,fname),'.vmp']);
