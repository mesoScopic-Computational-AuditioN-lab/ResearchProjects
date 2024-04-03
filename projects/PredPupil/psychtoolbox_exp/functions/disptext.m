function disptext( w, textstr, screenrect, fontsize, textfont, textcol, xydeviation)

% text settings
Screen( 'TextFont', w, textfont);
Screen( 'TextSize', w, fontsize);

% get text box
bbox        =   Screen( 'TextBounds', w, textstr);
textrect    =   CenterRect( bbox, screenrect);

% add deviation
if exist('xydeviation','var')
   textrect(1,1:2) = textrect(1,1:2) + xydeviation; 
end

% draw text
Screen( 'DrawText', w, textstr, textrect(1), textrect(2), textcol);

end

