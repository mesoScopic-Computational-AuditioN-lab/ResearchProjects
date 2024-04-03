function grad = fetch_grad(sub_id)

dirname = sprintf('/project/3018063.01/raw/sub-%03d/ses-meg01/meg/', sub_id);
dsdir = dir([dirname '*.ds']);
grad = ft_read_sens([dirname dsdir.name], 'senstype', 'meg');

end