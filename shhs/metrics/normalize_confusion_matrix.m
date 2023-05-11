%% test
% C = [3529 579 97 46 258;
%     458 1219 353 29 703;
%     346 1215 13118 1676 1222;
%     80 31 461 5003 16;
%     219 781 470 6 6235];

function normC = normalize_confusion_matrix(C)
    normC = zeros(size(C));
    for i = 1 : size(normC,1)
        normC(i,:) = 100*(C(i,:)/sum(C(i,:)));
    end
end