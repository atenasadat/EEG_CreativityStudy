function results = find_edf_with_task_names(root_folder)
    results = struct('file_path', {}, 'task_name', {});
    
    files = dir(fullfile(root_folder, '**', '*.edf')); 
    
    for i = 1:length(files)
        fname = lower(files(i).name);
        fpath = fullfile(files(i).folder, files(i).name);
 
        if contains(fpath,'pilot') || contains(fpath, 'Subject2')
            continue
        end

        if contains(fname, 'raw') && (contains(fname, 'treatment') || contains(fname, 'control'))
            % Extract task name A{number} or B{number}
            task_match = regexp(files(i).name, '[AB]\d+', 'match', 'once');
            
            if ~isempty(task_match)
                entry.file_path = fpath;
                entry.task_name = upper(task_match); 
                results(end+1) = entry;
            end
            elseif  contains(fname, 'raw') && contains(fname, 'baseline')
                entry.file_path = fpath;
                entry.task_name = 'baseline'; 
                results(end+1) = entry;
        elseif contains(fname,'raw') && contains(fname, 'creativity')
                entry.file_path = fpath;
                entry.task_name = 'creativityTests';
                results(end+1) = entry;
        end

     
    end
end


% data_root = '/Users/athenasaghi/Desktop/ALL DATA/ALLEEG/Raw/Baselines/';

data_root = '/Users/athenasaghi/Desktop/ALL DATA/ALLEEG/Raw/';
save_root = '/Users/athenasaghi/Desktop/CLEANDATA/';

files_with_tasks = find_edf_with_task_names(data_root);

for i = 1:length(files_with_tasks)
    fprintf('File: %s |Task %s\n', files_with_tasks(i).file_path, files_with_tasks(i).task_name);

    tokens = regexp(files_with_tasks(i).file_path, '(?:P|s)(\d+)', 'tokens');

    if ~isempty(tokens)
        pid = str2double(tokens{1}{1});
        fprintf('Extracted Pid = %d\n', pid);
    else
        warning('-----No Pid found in file path.');
    end

    singlefilecore(files_with_tasks(i).file_path,files_with_tasks(i).task_name, pid, data_root, save_root)
end

