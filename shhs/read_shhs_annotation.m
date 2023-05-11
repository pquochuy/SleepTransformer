%filename = './polysomnography/annotations-events-nsrr/shhs1/shhs1-200001-nsrr.xml';
function [stages] = read_shhs_annotation(filename)
    text = fileread(filename);

    expr = '<EventConcept>Recording Start Time</EventConcept>\r\n<Start>0</Start>';
    %\n<Start>0</Start>

    %expr = '[^\n]*fileread[^\n]*';
    matches = regexp(text, expr, 'match');

    assert(numel(matches) == 1);

    patterns_stages = ['<EventType>Stages.Stages</EventType>\r\n<EventConcept>[0-9a-zA-Z\s|]+</EventConcept>\r\n<Start>[0-9\.]+</Start>\r\n<Duration>[0-9\.]+</Duration>'];
    matches = regexp(text, patterns_stages, 'match');

    disp(matches{end})

    stages = [];
    starts = [];
    durations = [];
    for m = 1 : numel(matches)
        lines = splitlines(matches{m});
        stageline = lines{2};
        stage = str2double(stageline(end-15));
        startline = lines{3};
        start = str2double(startline(8 : end-8));
        durationline = lines{4};
        duration = str2double(durationline(11 : end-11));
        assert (mod(duration,30)== 0.)
        num_epoch = duration/30;
        stages = [stages; ones(num_epoch, 1)*stage];
        starts = [starts; start];
        durations = [durations; duration];
    end
    % last 'start' and 'duration' are still in mem
    % verify that we have not missed stuff..
    assert((start + duration)/30 == length(stages))
end