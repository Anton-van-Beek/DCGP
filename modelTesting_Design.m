clc; clear all; close all

%% Initiate objective
Prob_ID = 1;
DC_data = problems_database(Prob_ID);
Bayes_Type = 'Partial';                                                     % Partial, Full

f = DC_data.f;          nSamples = DC_data.nSamples; 
V = DC_data.V;          xTrain = DC_data.xTrain;
D = DC_data.D;          yTrain = DC_data.yTrain;
l = DC_data.l;          lTrain = DC_data.lTrain;
Sigma = DC_data.Sigma;  Bayes_nMcSamples = DC_data.Bayes_nMcSamples;
DC_Type = DC_data.DC_Type;


%% Model testing
nMacroTrial = 2;
nMicroTrial = 15*size(DC_data.xTrain,2);
opt_x_hat = zeros(4,size(DC_data.xTrain,2),nMacroTrial,nMicroTrial+1); 
opt_y_hat = zeros(4,nMacroTrial,nMicroTrial+1);
opt_y = zeros(4,nMacroTrial,nMicroTrial+1);
nMcSamples = 100;                                                           % Per dimension
nCand = nMcSamples*size(DC_data.xTrain,2);
xCand = lhsdesign(nCand,size(DC_data.xTrain,2)).*repmat((DC_data.xMax'-DC_data.xMin'),nCand,1)+repmat((DC_data.xMin'),nCand,1);

xT = zeros(size(DC_data.xTrain,1)+nMicroTrial,size(DC_data.xTrain,2));
xT(1:size(DC_data.xTrain,1),:) = xTrain;
yT = zeros(size(DC_data.xTrain,1)+nMicroTrial,1);
yT(1:size(DC_data.xTrain,1),:) = yTrain;
lT = -1*ones(size(DC_data.xTrain,1)+nMicroTrial,1);
lT(1:size(DC_data.xTrain,1),:) = l(xTrain);

q = 1 + max(lTrain);
%%
for i = 1:nMacroTrial
    %% Standard GP
    for j = 1:nMicroTrial
        EI = zeros(nMcSamples,1);
        fmin = min(yT(1:size(DC_data.xTrain,1)+j-1,:));
        model = FitGP([xT(1:size(DC_data.xTrain,1)+j-1,:),l(xT(1:size(DC_data.xTrain,1)+j-1,:))],yT(1:size(DC_data.xTrain,1)+j-1,:));
        [opt_y(1,i,j), opt_x_hat(1,:,i,j),opt_y_hat(1,i,j)] = robust_opt_GP(model,Sigma,xCand,nCand,l,f);
        for k = 1:nCand
            xCandSam = xCand(k,:) + mvnrnd(zeros(1,size(DC_data.xTrain,2)),Sigma,nCand);
            [pred, predMse] = PredictGP([xCandSam,l(xCandSam)],model, 'MSE_Flag', 'On');
            predStd = sqrt(sqrt(predMse.^2));
            d = fmin - pred;
            EI_sample = (pred - fmin).*normpdf(d./sqrt(predStd)) + sqrt(predStd).*normcdf(d./sqrt(predStd));
            EI(k,1) = mean(EI_sample.*double(sum([lt(repmat(DC_data.xMin',nCand,1), xCandSam),gt(repmat(DC_data.xMax',nCand,1), xCandSam)],2) == 2*size(xTrain,2)));
            if rem(k,10) == 0
                disp(strcat('Macro Trial:',num2str(i),'/',num2str(nMacroTrial),'. Micro Trial:',num2str(j),'/',num2str(nMicroTrial),'. Model type: GP. Completed: ',num2str(round(k/nCand*100)),'%'));
            end
        end
        [~, ind] = max(EI);
        xNew = xCand(ind,:);
        xT(size(DC_data.xTrain,1)+j,:) = xNew;
        yT(size(DC_data.xTrain,1)+j,:) = f(xNew);
    end
    [opt_y(1,i,j+1), opt_x_hat(1,:,i,j+1),opt_y_hat(1,i,j)] = robust_opt_GP(model,Sigma,xCand,nCand,l,f);
    
    %% Multi GP
    
    for j = 1:nMicroTrial
        EI = zeros(nMcSamples,1);
        fmin = min(yT(1:size(DC_data.xTrain,1)+j-1,:));
        if strcmp(DC_Type,{'Jump'})
            Q = q;
        else
            Q = size(q,2);
        end

        if 0 < Q
            if size(xT(lT==0,:),1) == 1
                model1 = FitGP([xT(lT==0,:);1.1*xT(lT==0,:)],[yT(lT==0,:);1.1*yT(lT==0,:)]);
            else
                model1 = FitGP(xT(lT==0,:),yT(lT==0,:));
            end
        end
        if 1 < Q
            if size(xT(lT==1,:),1) == 1
                model2 = FitGP([xT(lT==1,:);1.1*xT(lT==1,:)],[yT(lT==1,:);1.1*yT(lT==1,:)]);
            else
                model2 = FitGP(xT(lT==1,:),yT(lT==1,:));
            end
        end
        if 2 < Q
            if size(xT(lT==2,:),1) == 1
                model3 = FitGP([xT(lT==2,:);1.1*xT(lT==2,:)],[yT(lT==2,:);1.1*yT(lT==2,:)]);
            else
                model3 = FitGP(xT(lT==2,:),yT(lT==2,:));
            end
        end       
        if Q == 2
            [opt_y(2,i,j), opt_x_hat(2,:,i,j),opt_y_hat(2,i,j)] = robust_opt_twoGP(model1,model2,Sigma,xCand,nCand,l,f);
        elseif Q == 3
            [opt_y(2,i,j), opt_x_hat(2,:,i,j),opt_y_hat(2,i,j)] = robust_opt_threeGP(model1,model2,model3,Sigma,xCand,nCand,l,f);
        end
        for k = 1:nCand
            xCandSam = xCand(k,:) + mvnrnd(zeros(1,size(DC_data.xTrain,2)),Sigma,nCand);
            lTrainTemp = l(xCandSam);
            if Q == 2
                xPred1 = xCandSam(lTrainTemp == 0);
                [pred1, predMse1] = PredictGP(xPred1,model1, 'MSE_Flag', 'On');
                xPred2 = xCandSam(lTrainTemp == 1);
                [pred2, predMse2] = PredictGP(xPred2,model2, 'MSE_Flag', 'On');
                xCanSamShuf = [xPred1;xPred2];
                pred = [pred1;pred2];
                predMse = [predMse1;predMse2];
            elseif Q == 3
                xPred1 = xCandSam(lTrainTemp == 0,:);
                try
                    [pred1, predMse1] = PredictGP(xPred1,model1, 'MSE_Flag', 'On');
                catch
                    pred1 = [];
                    predMse1 = [];
                end
                xPred2 = xCandSam(lTrainTemp == 1,:);
                try
                    [pred2, predMse2] = PredictGP(xPred2,model2, 'MSE_Flag', 'On');
                catch
                    pred2 = [];
                    predMse2 = [];
                end
                xPred3 = xCandSam(lTrainTemp == 2,:);
                try
                    [pred3, predMse3] = PredictGP(xPred3,model3, 'MSE_Flag', 'On');
                catch
                    pred3 = [];
                    predMse3 = [];
                end
                xCanSamShuf = [xPred1;xPred2;xPred3];
                pred = [pred1;pred2;pred3];
                predMse = [predMse1;predMse2;predMse3];
            end
            predStd = sqrt(sqrt(predMse.^2));
            d = fmin - pred;
            EI_sample = (pred - fmin).*normpdf(d./sqrt(predStd)) + sqrt(predStd).*normcdf(d./sqrt(predStd));
            EI(k,1) = mean(EI_sample.*double(sum([lt(repmat(DC_data.xMin',nCand,1), xCandSam),gt(repmat(DC_data.xMax',nCand,1), xCandSam)],2) == 2*size(xTrain,2)));
            if rem(k,10) == 0
                disp(strcat('Macro Trial:',num2str(i),'/',num2str(nMacroTrial),'. Micro Trial:',num2str(j),'/',num2str(nMicroTrial),'. Model type: Multi-GP. Completed: ',num2str(round(k/nCand*100)),'%'));
            end
        end
        [~, ind] = max(EI);
        xNew = xCand(ind,:);
        xT(size(DC_data.xTrain,1)+j,:) = xNew;
        yT(size(DC_data.xTrain,1)+j,:) = f(xNew);
        lT(size(DC_data.xTrain,1)+j,:) = l(xNew);
    end
    if Q == 2
        [opt_y(2,i,j+1), opt_x_hat(2,:,i,j+1),opt_y_hat(2,i,j+1)] = robust_opt_twoGP(model1,model2,Sigma,xCand,nCand,l,f);
    elseif Q == 3
        [opt_y(2,i,j+1), opt_x_hat(2,:,i,j+1),opt_y_hat(2,i,j+1)] = robust_opt_threeGP(model1,model2,model3,Sigma,xCand,nCand,l,f);
    end
    %% DCGP Partial
    Bayes_Type = 'Partial'; 
    for j = 1:nMicroTrial
        EI = zeros(nMcSamples,1);
        fmin = min(yT(1:size(DC_data.xTrain,1)+j-1,:));

        if strcmp(DC_Type,'Jump')
            model = DCGP_fit(xT(1:size(DC_data.xTrain,1)+j-1,:),yT(1:size(DC_data.xTrain,1)+j-1,:),l(xT(1:size(DC_data.xTrain,1)+j-1,:)),V,D,'DC_Type',DC_Type,'l',l); 
        else
            model = DCGP_fit(xT(1:size(DC_data.xTrain,1)+j-1,:),yT(1:size(DC_data.xTrain,1)+j-1,:),lTrain,V,D,'DC_Type',DC_Type,'l',l);
        end
        [opt_y(3,i,j), opt_x_hat(3,:,i,j),opt_y_hat(3,i,j)] = robust_opt_DCGP(model,Sigma,xCand,nCand,l,f,Bayes_Type,Bayes_nMcSamples);
        for k = 1:nCand
            xCandSam = xCand(k,:) + mvnrnd(zeros(1,size(DC_data.xTrain,2)),Sigma,nCand);
            [pred, predMse] = Pred_DCGP(xCandSam,l(xCandSam),nCand,model,'Bayes_Type',Bayes_Type,'Bayes_nMcSamples',Bayes_nMcSamples);
            predStd = sqrt(sqrt(predMse.^2));
            d = fmin - pred;
            try
                EI_sample = (pred - fmin).*normpdf(d./sqrt(predStd)) + sqrt(predStd).*normcdf(d./sqrt(predStd));
            catch
                EI_sample = 0;
            end
            EI(k,1) = mean(EI_sample.*double(sum([lt(repmat(DC_data.xMin',nCand,1), xCandSam),gt(repmat(DC_data.xMax',nCand,1), xCandSam)],2) == 2*size(xTrain,2)));
            if rem(k,10) == 0
                disp(strcat('Macro Trial:',num2str(i),'/',num2str(nMacroTrial),'. Micro Trial:',num2str(j),'/',num2str(nMicroTrial),'. Model type: DCGP MLE. Completed: ',num2str(round(k/nCand*100)),'%'));
            end
        end
        [~, ind] = max(EI);
        xNew = xCand(ind,:);
        xT(size(DC_data.xTrain,1)+j,:) = xNew;
        yT(size(DC_data.xTrain,1)+j,:) = f(xNew);
    end
    [opt_y(3,i,j+1), opt_x_hat(3,:,i,j+1),opt_y_hat(3,i,j)] = robust_opt_DCGP(model,Sigma,xCand,nCand,l,f,Bayes_Type,Bayes_nMcSamples);
    
    %% DCGP Full Bayesian
    Bayes_Type = 'Full'; 
    for j = 1:nMicroTrial
        EI = zeros(nMcSamples,1);
        fmin = min(yT(1:size(DC_data.xTrain,1)+j-1,:));

        if strcmp(DC_Type,'Jump')
            model = DCGP_fit(xT(1:size(DC_data.xTrain,1)+j-1,:),yT(1:size(DC_data.xTrain,1)+j-1,:),l(xT(1:size(DC_data.xTrain,1)+j-1,:)),V,D,'DC_Type',DC_Type,'l',l); 
        else
            model = DCGP_fit(xT(1:size(DC_data.xTrain,1)+j-1,:),yT(1:size(DC_data.xTrain,1)+j-1,:),lTrain,V,D,'DC_Type',DC_Type,'l',l);
        end
        [opt_y(4,i,j), opt_x_hat(4,:,i,j),opt_y_hat(4,i,j)] = robust_opt_DCGP(model,Sigma,xCand,nCand,l,f,Bayes_Type,Bayes_nMcSamples);
        for k = 1:nCand
            xCandSam = xCand(k,:) + mvnrnd(zeros(1,size(DC_data.xTrain,2)),Sigma,nCand);
            [pred, predMse] = Pred_DCGP(xCandSam,l(xCandSam),nCand,model,'Bayes_Type',Bayes_Type,'Bayes_nMcSamples',Bayes_nMcSamples);
            predStd = sqrt(sqrt(predMse.^2));
            d = fmin - pred;
            try
                EI_sample = (pred - fmin).*normpdf(d./sqrt(predStd)) + sqrt(predStd).*normcdf(d./sqrt(predStd));
            catch
                EI_sample = 0;
            end
            EI(k,1) = mean(EI_sample.*double(sum([lt(repmat(DC_data.xMin',nCand,1), xCandSam),gt(repmat(DC_data.xMax',nCand,1), xCandSam)],2) == 2*size(xTrain,2)));

            if rem(k,10) == 0
                disp(strcat('Macro Trial:',num2str(i),'/',num2str(nMacroTrial),'. Micro Trial:',num2str(j),'/',num2str(nMicroTrial),'. Model type: DCGP Bayes. Completed: ',num2str(round(k/nCand*100)),'%'));
            end

        end
        
        [~, ind] = max(EI);
        xNew = xCand(ind,:);
        xT(size(DC_data.xTrain,1)+j,:) = xNew;
        yT(size(DC_data.xTrain,1)+j,:) = f(xNew);
    end
    [opt_y(4,i,j+1), opt_x_hat(4,:,i,j+1),opt_y_hat(4,i,j)] = robust_opt_DCGP(model,Sigma,xCand,nCand,l,f,Bayes_Type,Bayes_nMcSamples);

    rng("shuffle")

    DC_data = problems_database(Prob_ID);
    Bayes_Type = 'Partial';                                                     % Partial, Full
    
    f = DC_data.f;          nSamples = DC_data.nSamples; 
    V = DC_data.V;          xTrain = DC_data.xTrain;
    D = DC_data.D;          yTrain = DC_data.yTrain;
    l = DC_data.l;          lTrain = DC_data.lTrain;
    Sigma = DC_data.Sigma;  Bayes_nMcSamples = DC_data.Bayes_nMcSamples;
    DC_Type = DC_data.DC_Type;
    
    xT = zeros(size(DC_data.xTrain,1)+nMicroTrial,size(DC_data.xTrain,2));
    xT(1:size(DC_data.xTrain,1),:) = xTrain;
    yT = zeros(size(DC_data.xTrain,1)+nMicroTrial,1);
    yT(1:size(DC_data.xTrain,1),:) = yTrain;
    lT = -1*ones(size(DC_data.xTrain,1)+nMicroTrial,1);
    lT(1:size(DC_data.xTrain,1),:) = l(xTrain);
end

figure(1); set(gcf,'position',[50,50,600,500]); hold on
plot(0:nMicroTrial,mean(squeeze(opt_y(1,:,:))),'k','LineWidth',2);
plot(0:nMicroTrial,mean(squeeze(opt_y(2,:,:))),'b','LineWidth',2);
plot(0:nMicroTrial,mean(squeeze(opt_y(3,:,:))),'r','LineWidth',2);
plot(0:nMicroTrial,mean(squeeze(opt_y(4,:,:))),'g','LineWidth',2);
legend('GP','Multi-GP','DCGP-P','DCGP-B','NumColumns',4,'Location','northoutside')
xlabel('Number of iterations','Interpreter','latex')
ylabel('Approximated optimum','Interpreter','latex')
box on
set(gca,'linewidth',1.5,'FontSize',15)
Data = struct('opy_y',opt_y,'opt_x_hat',opt_x_hat,'opt_y_hat',opt_y_hat,'Prob_ID',Prob_ID);
save(strcat('Design-ProbID',num2str(Prob_ID),'-optimization.mat'),'Data')


