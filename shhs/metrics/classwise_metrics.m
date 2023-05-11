function [eval_metric] = classwise_metrics(normC)

    Ncat = size(normC,1);
    
    eval_metric = zeros(Ncat, 4);
    
    for cl = 1 : Ncat
        C_cl = zeros(2,2);
        
        pos = cl;
        neg = [1:Ncat];
        neg(cl) = [];
        
        p = sum(normC(pos,:));
        n = sum(sum(normC(neg,:)));
        N = p + n;
        
        tp = sum(normC(pos,pos));
        tn = sum(sum(normC(neg,neg)));
        fp = n - tn;
        fn = p - tp;
        
        C_cl(1,1) = tp;
        C_cl(1,2) = fn;
        C_cl(2,1) = fp;
        C_cl(2,2) = tn;
        
        normC_cl = normalize_confusion_matrix(C_cl);
        
        tp = normC_cl(1,1);
        fn = normC_cl(1,2);
        fp = normC_cl(2,1);
        tn = normC_cl(2,2);
        
        % precision
        precision = 100*tp/(tp + fp);
        % sensitivity (recall)
        sensitivity = 100*tp/(tp+fn);
        recall = sensitivity;
        % F1
        recall = sensitivity;
        F1 = 2*((precision*recall)/(precision + recall));
        % accuracy
        accuracy = (tp+tn)/(tp+tn+fp+fn);
        eval_metric(cl,1) = precision;
        eval_metric(cl,2) = sensitivity;
        eval_metric(cl,3) = F1;
        eval_metric(cl,4) = accuracy;
    end
end