function multilinetext( w, multi, screenrect, fontsize, textfont, textcol, ofset, halffont)

% text settings
Screen( 'TextFont', w, textfont);

% get text box
for i=1:length(multi)
    Screen( 'TextSize', w, fontsize);
    ofsets = ofset*fontsize;
    if ismember(i, halffont); Screen( 'TextSize', w, fontsize/1.8); ofsets = ofsets/1.8; end
    bbox        =   Screen( 'TextBounds', w, multi{i});
    textrect    =   CenterRect( bbox, screenrect + [0 ((i-1)*ofsets)-(0.5*length(multi)*ofsets) 0 ((i-1)*ofsets)-(0.5*length(multi)*ofsets)]);
    Screen( 'DrawText', w, multi{i}, textrect(1), textrect(2), textcol);
end    

end
