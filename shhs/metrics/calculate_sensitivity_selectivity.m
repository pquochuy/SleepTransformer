function [sensitivity, selectivity]  = calculate_sensitivity_selectivity(y, yhat)
    cat = unique(y);
    sensitivity = zeros(numel(cat),1);
    selectivity = zeros(numel(cat),1);
    for cl = 1 : numel(cat)
        
        % sensitivity
        ind = (yhat == cl);
        true_det = sum(y(ind) == cl);
        num_ref = sum(y == cl);
        num_det = sum(ind);
        sensitivity(cl) = true_det/num_ref;
        selectivity(cl) = true_det/num_det;
    end
end