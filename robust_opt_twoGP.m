function [y_mu, x_mu_hat,y_mu_hat] = robust_opt_twoGP(model1,model2,Sigma,xCand,nCand,l,f)

pred_mu = zeros(nCand,1);
for k = 1:nCand
    xCandSam = xCand(k,:) + mvnrnd(zeros(1,size(model1.XTN,2)),Sigma,nCand);
    lTrain = l(xCandSam);
    pred_mu(k,1) = mean([PredictGP(xCandSam(lTrain == 0),model1);PredictGP(xCandSam(lTrain == 1),model2)]);
end

[y_mu_hat, ind] = min(pred_mu);
x_mu_hat = xCand(ind,:);

x_muSam = x_mu_hat + mvnrnd(zeros(1,size(model1.XTN,2)),Sigma,nCand);
y_mu = mean(f(x_muSam));

end