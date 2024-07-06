function [y_mu, x_mu_hat,y_mu_hat] = robust_opt_DCGP(model,Sigma,xCand,nCand,l,f,Bayes_Type,Bayes_nMcSamples)
pred_mu = zeros(nCand,1);
parfor k = 1:nCand
    xCandSam = xCand(k,:) + mvnrnd(zeros(1,size(model.xn,2)),Sigma,nCand);
    pred_mu(k,:) = mean(Pred_DCGP(xCandSam,l(xCandSam),nCand,model,'Bayes_Type',Bayes_Type,'Bayes_nMcSamples',Bayes_nMcSamples));
end

[y_mu_hat, ind] = min(pred_mu);
x_mu_hat = xCand(ind,:);

x_muSam = x_mu_hat + mvnrnd(zeros(1,size(model.xn,2)),Sigma,nCand);
y_mu = mean(f(x_muSam));

end