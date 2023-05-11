function [f1, sensitivity, specificity]  = litis_class_wise_f1(y, yhat)
    cat = unique(y);
    f1 = zeros(numel(cat),1);
    sensitivity = zeros(numel(cat),1);
    specificity = zeros(numel(cat),1);
    for cl = 1 : numel(cat)
        %ind = (y == cl);
        %f1(cl) = sum(y(ind) == yhat(ind))/sum(ind);
        [f1(cl), sensitivity(cl), specificity(cl)]  = computeF1(y,yhat,cl);
    end
end

function [fscore, sensitivity, specificity] = computeF1(y,yhat,class)
    ind = (y == class);
    y(~ind) = 0;
    y(ind) = 1;
    
    ind = (yhat == class);
    yhat(~ind) = 0;
    yhat(ind) = 1;

%     class = unique(y);
%     y = ones(size(y));
%     ind = (yhat == class);
%     yhat(ind) = 1;
%     yhat(~ind) = 0;
    EVAL = Evaluate(y,yhat);
    fscore = EVAL(6);
    sensitivity = EVAL(2);
    specificity = EVAL(3);
end