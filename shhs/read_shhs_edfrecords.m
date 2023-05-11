function [signal, fs] = read_shhs_edfrecords(filename, channel)
    [header, edf] = edfread(filename,'targetSignals',channel);
    assert(length(header.samples) == numel(channel));
    
    fs = unique(header.samples);
    assert(length(fs) == 1);
    
    if(length(channel) > 1)
        signal = diff(edf)';
    else
        signal = edf';
    end
end

