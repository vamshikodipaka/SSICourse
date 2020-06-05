function regs_param = findRegPara( data, labels, lambda )
% this function is used to get parameters by optmising the cost function
% inputs:--------------
%     data                data with feaures
%     labels              labels of data
%     lambda              regularization parameters         
% outputs: --------------
%     regs_param        : we get regression parameters 
% ///////////////////////////////////////////////////

regs_param = zeros(size(data,2),1); % to intialise the parameters
%  options = optimoptions('fminunc','Display','iter','GradObj','on','MaxIter',400);
options = optimoptions('fminunc','GradObj','on','MaxIter',400);
% This function is to minimise cost function of logistic regression and find
% parameters

[regs_param, ~] = fminunc(@(regs_para)costFunction_regu(data, labels, regs_param, lambda), regs_param, options);

end

