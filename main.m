clc; clear all; close all

%% Initiate objective
Prob_ID = 1;
DC_data = problems_database(Prob_ID);
Bayes_Type = 'Partial';                                                     % Partial, Full

f = DC_data.f;          nSamples = DC_data.nSamples;
V = DC_data.V;          xTrain = DC_data.xTrain;
D = DC_data.D;          yTrain = DC_data.yTrain;
l = DC_data.l;          lTrain = DC_data.lTrain;

Bayes_nMcSamples = DC_data.Bayes_nMcSamples;
DC_Type = DC_data.DC_Type;


%% Train model
model = DCGP_fit(xTrain,yTrain,lTrain,V,D,'DC_Type',DC_Type,'l',l);

%% Predict response

if model.d == 1
    nPlot = 500;
    xPred = linspace(0,1,nPlot)';
    lPred = l(xPred);
    [yPred,yStd] = Pred_DCGP(xPred,lPred,nPlot,model,'Bayes_Type',Bayes_Type,'Bayes_nMcSamples',Bayes_nMcSamples);
    figure; set(gcf,'position',[50,50,600,500]);
    h1 = fill([xPred(1:250,1)',fliplr(xPred(1:250,1)')],[yPred(1:250,1)'+1.96*yStd(1:250,1)',fliplr(yPred(1:250,1)'-1.96*yStd(1:250,1)')],'B','FaceAlpha',0.2,'EdgeAlpha',0); hold on
    h2 = plot(xPred(1:250,1),yPred(1:250,1),'-b','linewidth',1.5);
    h3 = plot(xPred(1:250,1),f(xPred(1:250,1)),'-k','linewidth',1.5); 
    h4 = scatter(xTrain,yTrain,50,'filled','blue');
    h5 = fill([xPred(251:end,1)',fliplr(xPred(251:end,1)')],[yPred(251:end,1)'+1.96*yStd(251:end,1)',fliplr(yPred(251:end,1)'-1.96*yStd(251:end,1)')],'B','FaceAlpha',0.2,'EdgeAlpha',0); hold on
    h6 = plot(xPred(251:end,1),yPred(251:end,1),'-b','linewidth',1.5);
    h7 = plot(xPred(251:end,1),f(xPred(251:end,1)),'-k','linewidth',1.5); 
    axis([0 1 -15 10])
    legend([h1,h2,h3,h4],'$95\%$ PI','$E[\hat{f}(x)]$','$f(x)$','$\textbf{D}$','NumColumns',2,'Location','southwest','Interpreter','latex')
    ylabel('Output: $y$','Interpreter','latex')
    xlabel('Input: $x$','Interpreter','latex')
    box on; grid on
    set(gca,'linewidth',1.5,'FontSize',15)

elseif model.d == 2
    prec = 251;
    [X1,X2] = meshgrid(linspace(-5,5,prec),linspace(-5,5,prec));
    yPred = zeros(prec,prec);
    yStd = zeros(prec,prec);
    for j = 1:prec
        lPred = l([X1(:,j),X2(:,j)]);
        [yPred(:,j),yStd(:,j)] = Pred_DCGP([X1(:,j),X2(:,j)],lPred,prec,model,'Bayes_Type',Bayes_Type,'Bayes_nMcSamples',Bayes_nMcSamples);
    end
    surf(X1,X2,yPred,'linestyle','none');
end