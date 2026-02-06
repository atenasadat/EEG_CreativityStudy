

%% CoreCleaning.m - EEG Data Preprocessing Pipeline
% 
% DESCRIPTION:
%   This MATLAB script processes raw EEG data for the EEG Creativity Study.
%   It performs signal cleaning, channel renaming, filtering, and event detection
%   from CSV annotation files.
%
% MAIN WORKFLOW:
%   1. Load raw EEG data (BioSig format)
%   2. Parse trigger events from CSV annotations and original EEG trigger channel
%   3. Rename channels to standard nomenclature (T3→T7, T4→T8, etc.)
%   4. Remove unnecessary channels (reference channels, X channels)
%   5. Apply channel location lookup and re-referencing
%   6. Apply bandpass filter (0.5-100 Hz)
%   7. Save processed EEG dataset
%
% DEPENDENCIES:
%   - EEGLAB toolbox 
%   - Signal Processing Toolbox
%   - BioSig file format support
%
% =========================================================================

function CoreCleaning(input_file, save_root, pid, taskname, condition, root_file)
   
    % INPUTS:
    %   input_file (char)    - Path to raw EEG data file (relative to root_file)
    %   save_root (char)     - Directory path where processed .set file will be saved
    %   pid (int)            - Participant ID number
    %   taskname (char)      - Name of the task (e.g., 'Divergent', 'Convergent')
    %   condition (char)     - Experimental condition (e.g., 'Rest', 'Music')
    %   root_file (char)     - Base directory path for input file resolution
    %
    % OUTPUTS:
    %   Saves processed EEG dataset as: Filter100_P<pid>_<condition>_<taskname>.set
    %
    % EXAMPLE:
    %   CoreCleaning('rawdata.bdf', '/output/', 1, 'Divergent', 'Music', '/data/')
    
    EEG = preprocess_eeg(input_file, root_file);
    save_filename = ['Filter100_P' num2str(pid) '_' condition '_' taskname '.set'];
    pop_saveset(EEG, 'filename', save_filename, 'filepath', save_root);
end

function EEG = preprocess_eeg(input_file, root_file)
    [ALLEEG, EEG, CURRENTSET, ~] = eeglab('nogui');
    EEG = pop_biosig(input_file);
    [ALLEEG, EEG, CURRENTSET] = eeg_store(ALLEEG, EEG, 1);
  
    trigger_idx = find(strcmp({EEG.chanlocs.labels}, 'Trigger'));
   
    trigger_signal = EEG.data(trigger_idx, :);
    timestamps_sec = (0:EEG.pnts - 1) / EEG.srate;
    trigger_from_comments = zeros(size(trigger_signal));
    [folder, base_name, ~] = fileparts(fullfile(root_file, input_file));
    csv_file = fullfile(folder, [base_name '.csv']);
    if isfile(csv_file)
        opts = detectImportOptions(csv_file);
        opts = setvartype(opts, 'Comments', 'string');
        T = readtable(csv_file, opts);
        if ismember('Time', T.Properties.VariableNames)
            comment_times = T.Time;
        else
            error('CSV must contain a "Time" column with timestamps (in seconds).');
        end
        if ismember('Comments', T.Properties.VariableNames)
            comment_strings = string(T.Comments);
            trigger_map = containers.Map({'start prompting', 'aha moment'}, [16, 4]);
            for k_map = keys(trigger_map)
                keyword = k_map{1};
                trig_val = trigger_map(keyword);
                match_idx = contains(lower(comment_strings), keyword);
                event_times = comment_times(match_idx);
                for j = 1:length(event_times)
                    [~, eeg_idx] = min(abs(timestamps_sec - event_times(j)));
                    trigger_from_comments(eeg_idx) = trig_val;
                end
            end
        else
            warning('"Comments" column not found in CSV: %s', csv_file);
        end
    else
        warning('CSV annotation file not found: %s', csv_file);
    end
    final_trigger_signal = max(trigger_signal, trigger_from_comments);
    new_eeg_events = struct('type', {}, 'latency', {}, 'duration', {}, 'channel', {});
    min_inter_event_interval_samples = round(0.05 * EEG.srate);
    previous_value = 0;
    previous_onset_latency = -inf;
    for i = 1:length(final_trigger_signal)
        current_value = final_trigger_signal(i);
        if current_value ~= 0 && (current_value ~= previous_value || ...
                                  (i - previous_onset_latency) > min_inter_event_interval_samples)
            new_event_entry.type = num2str(current_value);
            new_event_entry.latency = i;
            new_event_entry.duration = 0;
            new_event_entry.channel = trigger_idx;
            new_eeg_events(end+1) = new_event_entry;
            previous_onset_latency = i;
        end
        previous_value = current_value;
    end
    EEG.event = new_eeg_events;
    fprintf('Replaced EEG.event with %d events derived from final_trigger_signal.\n', length(EEG.event));

    display_eeg_events_detail(EEG.event, 'EEG.event Structure (Before Cleaning/ICA)');

    EEG = eeg_checkset(EEG, 'eventconsistency');
    EEG = pop_select(EEG, 'nochannel', trigger_idx);
    disp('Removed the original "Trigger" data channel from EEG.data.');
    for i = 1:length(EEG.chanlocs)
        EEG.chanlocs(i).labels = regexprep(EEG.chanlocs(i).labels, '[-:].*', '');
    end

   % RAW data renamings 
    rename_map = containers.Map({'T3','T4','T5','T6'},{'T7','T8','P7','P8'});
    for i = 1:length(EEG.chanlocs)
        lbl = EEG.chanlocs(i).labels;
        if isKey(rename_map, lbl)
            EEG.chanlocs(i).labels = rename_map(lbl);
        end
    end

    EEG = pop_select(EEG, 'nochannel', {'CM','X3:S1-Pz','X2:S2-Pz','X1:S3-Pz','X1','X2','X3'});
    [ALLEEG, EEG] = eeg_store(ALLEEG, EEG, CURRENTSET);
    for i = 1:length(EEG.chanlocs)
        EEG.chanlocs(i).labels = regexprep(EEG.chanlocs(i).labels, '-Pz$', '');
    end
    target_labels = {'Pz'};
    ground_labels = {'A1','A2'};
    for i = 1:length(EEG.chanlocs)
        if ismember(EEG.chanlocs(i).labels, target_labels)
            EEG.chanlocs(i).type = 'REF';
        end
        if ismember(EEG.chanlocs(i).labels, ground_labels)
            EEG.chanlocs(i).type = 'GND';
        end
    end


    EEG = pop_chanedit(EEG, 'lookup', '/Users/athenasaghi/Downloads/eeglab2025.0.0/sample_locs/Standard-10-20-Cap25.locs');
    EEG = pop_reref(EEG, []);


    EEG = pop_eegfiltnew(EEG, 0.5, 100);
    % EEG = pop_eegfiltnew(EEG, 0.5, 0);
    % [ALLEEG, EEG] = eeg_store(ALLEEG, EEG, CURRENTSET);
    % EEG = pop_eegfiltnew(EEG, 59, 61, [], 1);
    [ALLEEG, EEG] = eeg_store(ALLEEG, EEG, CURRENTSET);
    
    
    %% Bad channel rejection 
    % 
    % [EEG, indelec] = pop_rejchan(EEG, ...
    %     'threshold', 5, ...          % reject if > 5 SDs from mean
    %     'norm', 'on', ...
    %     'measure', 'kurt');    
    % 
    % if isempty(indelec)
    %     disp('No bad channels detected.');
    % else
    % fprintf('\n Bad channels detected and removed:\n');
    %     disp(indelec);
    % end

    % [ALLEEG, EEG] = eeg_store(ALLEEG, EEG, CURRENTSET);

    % === Run ICA ===
    % EEG = pop_runica(EEG, 'extended', 1);
    % pause;
    % pop_selectcomps(EEG, 1:size(EEG.icaweights,1)); % GUI only

    % % display_eeg_events_detail(EEG.event, 'event Structure (After Cleaning/ICA)');
    % % 
    % % count_event_occurrences(EEG.event);
end

function display_eeg_events_detail(events, title_str)
    fprintf('\n--- %s ---\n', title_str);
    if isempty(events)
        fprintf('No events found.\n');
        return;
    end
    fprintf('Type\tLatency\n');
    fprintf('----\t-------\n');
    for i = 1:length(events)
        fprintf('%s\t%d\n', events(i).type, events(i).latency);
    end
    fprintf('----------------------\n');
end

function count_event_occurrences(events)
    if isempty(events)
        disp('No events found in EEG.event.');
        return;
    end
    event_types = {events.type};
    unique_types = unique(event_types);
    for i = 1:length(unique_types)
        current_type = unique_types{i};
        count = sum(strcmp(event_types, current_type));
        fprintf('  Event Type "%s": %d occurrences\n', current_type, count);
    end
end