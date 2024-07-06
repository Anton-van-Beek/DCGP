function [y_mu, x_mu_hat,y_mu_hat] = robust_opt_GP(model,Sigma,xCand,nCand,l,f)
pred_mu = zeros(nCand,1);
for k = 1:nCand
    xCandSam = xCand(k,:) + mvnrnd(zeros(1,size(model.XTN,2)-1),Sigma,nCand);
    pred_mu(k,:) = mean(PredictGP([xCandSam,l(xCandSam)],model));
end

[y_mu_hat, ind] = min(pred_mu);
x_mu_hat = xCand(ind,:);

x_muSam = x_mu_hat + mvnrnd(zeros(1,size(model.XTN,2)-1),Sigma,nCand);
y_mu = mean(f(x_muSam));

end