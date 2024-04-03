function [bull_tex,bullmat] = get_bull_tex(w,colour,fuse_cont,fixrads,trans)
%% [bull_tex[,bull_mat]]= get_bull_tex(bull_dim,w [,colour][,fuse_cont][,colour]).m
% This function returns white bulls eye texture. Optionally, the eye
% take on a different colour (e.g. feedback).
%
% INPUT:
% -w (psychtoolbox window pointer
% -colour (optional, colour of middle stop - default: [1 1 1])
% -fixrads (optional, shape width of lines: mid, inner, outer)
% -fuse_cont (optional colour within ring)
%
% If the resolution is low you can try to blow up fixrads e.g. [30 60 30]
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
% Micha, April 2018
if nargin<5
    trans=true;
    if nargin<4
        fixrads= [3 6 3]; % midpoint, inner_ring, outer_ring size
        if nargin<3
            fuse_cont=0.5;
            if nargin<2
                colour=[1 1 1];
            end
        end
    end
end

% to do: parameterise this one slightly more
if numel(fixrads)>2
    bullprof=[ones(1,fixrads(1)) ones(1,fixrads(2))*fuse_cont ... % make profile
        ones(1,fixrads(3)) ...
        fuse_cont*ones(1,round(0.5*(sqrt(2)*2*sum(fixrads)-2*sum(fixrads))))];
else
    bullprof=[ones(1,fixrads(1)) ones(1,fixrads(2))*fuse_cont ... % make profile
        fuse_cont*ones(1,round(0.5*(sqrt(2)*2*sum(fixrads)-2*sum(fixrads))))];
end

% make a second texture of the same size that is fully transparent
b_eyeprof=zeros(size(bullprof));

% ... except in the centre
b_eyeprof(1:fixrads(1))=1;

% turn profiles into arrays
bullmat=arraytorad(2*sum(fixrads)+2,bullprof,1);
b_eyemat=arraytorad(2*sum(fixrads)+2,b_eyeprof,1); % profile for eye

% set only red/green plus opacitylayers for corr/incorr
bull_fillers=zeros(size(b_eyemat));

% try to use eye_as logical index
thresh_val=min(b_eyeprof);
b_eye_i=(b_eyemat)>thresh_val;

% adjust transparency
transbull=bullmat;transbull(transbull<.6)=0;
bullmat=repmat(bullmat,1,1,4);
[x,y]=meshgrid(size(bullmat));

if trans==true,
    bullmat(:,:,4)=transbull;
else
    trans_circ=get_circle(size(bullmat,1));
    bullmat(:,:,4)=trans_circ(1:size(bullmat,1),1:size(bullmat,2));
end

% set colour of the eye
bullmat = set_colour(bullmat,b_eye_i,colour);

% make texture
bull_tex=Screen('MakeTexture',w,bullmat.*255); % bullseye



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% nested functionss
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% set colour
% ~~~~~~~~~~
    function mat_out = set_colour(bullmat,b_eye_i,colour)
        mat_out=bullmat;
        for i_c=1:numel(colour)
            this_mat_out=mat_out(:,:,i_c);
            this_mat_out(b_eye_i)=colour(i_c);
            mat_out(:,:,i_c)=this_mat_out;
        end
    end


% arraytorad
% ~~~~~~~~~~
    function radframe=arraytorad(framesize,array,interpol)
        
        % Creates a 2D matrix, containing a circle with diameter "framesize" and
        % profile "array". We interpolate values to make 'em smoother
        %
        maxhypo=ceil(sqrt((framesize/2)^2+(framesize/2)^2));
        if maxhypo>size(array,2);
            array=[array (ones(1,maxhypo-length(array))*array(end))];
        end
        array=[array(1) array];
        
        % compute hypothenusae
        [xvals yvals] = meshgrid(1:framesize);
        hypos = sqrt((xvals-.5*framesize-.5).^2+(yvals-.5*framesize-.5).^2);
        
        % project pixel values from array along hypothenusae
        radframe=ones(size(hypos));
        if nargin<3||interpol==0;
            hypos=round(hypos);
            radframe(hypos==hypos)=array(hypos+1);
        else % if linear interpolation
            hyposdec=hypos(:,:)-floor(hypos(:,:));
            radframe(ceil(hypos)==ceil(hypos))=array(floor(hypos)+1)+...
                (array(ceil(hypos)+1)-array(floor(hypos)+1)).*(hyposdec);
        end
    end

    function [circle] = get_circle(dim)
        %% draw circular aperture
        texsize=dim/2; % double to keep some nice resolution
        [x,y]=meshgrid(-texsize:texsize, -texsize:texsize);
        circle = (x.^2 + y.^2 <= (texsize)^2);
    end


end

