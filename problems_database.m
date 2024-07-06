function DC_data = problems_database(Prob_ID)
addpath 'C:\Users\Avanb\OneDrive\Documents\UCD\My Research\2023 Discontinious functions\TestFunctions'

if Prob_ID == 1
    f = @(x) (6.*x - 2).^2.*sin(12.*x-4) + (x>0.5)*5;
    V = [];
    D = [];
    l = @(x) double(x>0.5);  
    Bayes_nMcSamples = 100; 
    nSamples = 5;
    xTrain = lhsdesign(nSamples,1);
    %xTrain = [0.06;0.2;0.34;0.499;0.61;0.71;0.83;0.93];
    xMin = 0;   xMax = 1;
    yTrain = f(xTrain);
    lTrain = l(xTrain);
    DC_Type = 'Jump';                                                       % Jump, NonDif, Jump_NonDif 
    Sigma = [0.0005];
    DC_data = struct('f',f,'l',l,'V',V,'D',D,'DC_Type',DC_Type,...
        'Bayes_nMcSamples',Bayes_nMcSamples,'nSamples',nSamples,...
        'xTrain',xTrain,'yTrain',yTrain,'lTrain',lTrain,...
        'xMin',xMin,'xMax',xMax,'Sigma',Sigma);

%     x = linspace(0,1,251);
%     plot(x,f(x));
elseif Prob_ID == 2
    %%
    f = @(x) (6.*x - 2).^2.*sin(12.*x-4) - (x>0.8).*(x-0.8).*100;
    V = [1];
    D = [0.8];
    l = @(x) double(x>0.8);
    lTrain = [V,D];
    Bayes_nMcSamples = 100; 
    nSamples = 12;
    xTrain = lhsdesign(nSamples,1);
    xMin = 0;   xMax = 1;
    yTrain = f(xTrain);
    DC_Type = 'NonDif';                                                     % Jump, NonDif, Jump_NonDif 
    Sigma = [0.0005];
    DC_data = struct('f',f,'l',l,'V',V,'D',D,'DC_Type',DC_Type,...
        'Bayes_nMcSamples',Bayes_nMcSamples,'nSamples',nSamples,...
        'xTrain',xTrain,'yTrain',yTrain,'lTrain',lTrain,...
        'xMin',xMin,'xMax',xMax,'Sigma',Sigma);

%     x = linspace(0,1,251);
%     plot(x,f(x));
elseif Prob_ID == 3
    f = @(x) (6.*x - 2).^2.*sin(12.*x-4) - (x>0.5).*((x-0.5).*50 - 19);
    V = [1];
    D = [0.5];
    l = @(x) double(x>0.5);
    lTrain = [V,D];
    Bayes_nMcSamples = 100; 
    nSamples = 10;
    xTrain = lhsdesign(nSamples,1);
    xMin = 0;   xMax = 1;
    yTrain = f(xTrain);
    DC_Type = 'Jump_NonDif';                                                % Jump, NonDif, Jump_NonDif 
    Sigma = [0.0005];
    DC_data = struct('f',f,'l',l,'V',V,'D',D,'DC_Type',DC_Type,...
        'Bayes_nMcSamples',Bayes_nMcSamples,'nSamples',nSamples,...
        'xTrain',xTrain,'yTrain',yTrain,'lTrain',lTrain,...
        'xMin',xMin,'xMax',xMax,'Sigma',Sigma);

%     x = linspace(0,1,251);
%     plot(x,f(x));
elseif Prob_ID == 4
    %%
    f = @(x) griewank_JUMP(x);
    V = [];
    D = [];
    l = @(x) griewank_JUMP_l(x);
    Bayes_nMcSamples = 100; 
    nSamples = 40;
    lTrain = 0;
    while size(unique(lTrain),1) < 3
        xTrain = (lhsdesign(nSamples,2)-0.5).*10;
        lTrain = l(xTrain);
    end
    xMin = [-5;-5];   xMax = [5;5];
    yTrain = zeros(nSamples,1);
    for i = 1:nSamples
        yTrain(i,:) = f(xTrain(i,:));
    end
    DC_Type = 'Jump';                                                       % Jump, NonDif, Jump_NonDif 
    Sigma = [0.0025,0;0,0.0005];
    DC_data = struct('f',f,'l',l,'V',V,'D',D,'DC_Type',DC_Type,...
        'Bayes_nMcSamples',Bayes_nMcSamples,'nSamples',nSamples,...
        'xTrain',xTrain,'yTrain',yTrain,'lTrain',lTrain,...
        'xMin',xMin,'xMax',xMax,'Sigma',Sigma);
%     prec = 1001;
%     [X1,X2] = meshgrid(linspace(-5,5,prec),linspace(-5,5,prec));
%     Y = zeros(prec,prec);
%     for i = 1:prec
%         for j = 1:prec
%             Y(i,j) = f([X1(i,j),X2(i,j)]);
%         end
%     end
%     surf(X1,X2,Y,'linestyle','none');


elseif Prob_ID == 5
    %%
    V = [1.5,1;1,1.8];
    D = [-3;4];
    q = 2;
    nd = 2;
    beta = [0;2;-1;-1;-1.5];        % beta = [0;-2;1;1;2];
    f = @(x) griewank_NDF(x,V,D,q,nd,beta);
    l = @(x) griewank_NDF_l(x,V,D,q,nd);
    Bayes_nMcSamples = 100; 
    nSamples = 40;
    lCheck = 0;
    while size(unique(lCheck),1) < 3
        xTrain = (lhsdesign(nSamples,2)-0.5).*10;
        lCheck = l(xTrain);
    end
    xMin = [-5;-5];   xMax = [5;5];
    lTrain = [V,D];
    yTrain = zeros(nSamples,1);
    for i = 1:nSamples
        yTrain(i,:) = f(xTrain(i,:));
    end
    DC_Type = 'NonDif';                                                       % Jump, NonDif, Jump_NonDif 
    Sigma = [0.0025,0;0,0.0005];
    DC_data = struct('f',f,'l',l,'V',V,'D',D,'DC_Type',DC_Type,...
        'Bayes_nMcSamples',Bayes_nMcSamples,'nSamples',nSamples,...
        'xTrain',xTrain,'yTrain',yTrain,'lTrain',lTrain,...
        'xMin',xMin,'xMax',xMax,'Sigma',Sigma);

%%
%     figure
%     prec= 500;
%     [X1,X2] = meshgrid(linspace(-5,5,prec),linspace(-5,5,prec));
%     Y = zeros(prec,prec);
%     for i = 1:prec
%         Y(i,:) = f([X1(i,:)',X2(i,:)']);
%     end
%     surf(X1,X2,Y,'linestyle','none');    
%     
%     close all
%     figure(1); set(gcf,'position',[50,50,500,500]); hold on
%     box on
%     surf(X1,X2,Y,'linestyle','none');    
%     pbaspect([1, 1, 1]);
%     view(0,90)
%     set(gca,'YTickLabel',[]);
%     set(gca,'XTickLabel',[]);
%     xlabel('Input 1: $x_1$','Interpreter','latex')
%     ylabel('Input 2: $x_2$','Interpreter','latex')
%     set(gca,'linewidth',1.5,'FontSize',15)
% 
%     annotation('textbox',[.2 .1 .3 .3],'String','$\xi_1$','FitBoxToText','on','Interpreter','latex','BackgroundColor','white','FontSize',15);
%     annotation('textbox',[.55 .2 .3 .3],'String','$\xi_2$','FitBoxToText','on','Interpreter','latex','BackgroundColor','white','FontSize',15);
%     annotation('textbox',[.65 .5 .3 .3],'String','$\xi_3$','FitBoxToText','on','Interpreter','latex','BackgroundColor','white','FontSize',15);
%     print(gcf,'JUMP_patches.png','-dpng','-r1200');
% 
%     prec = 50;
%     [X1,X2] = meshgrid(linspace(-5,5,prec),linspace(-5,5,prec));
%     x = [reshape(X1,prec^2,1),reshape(X2,prec^2,1)];
%     v = V;
%     d = D;
% 
%     m = ones(1,1+q*nd);
%     M = ones(prec^2,1+q*nd);
% 
%     for i = 1:prec^2
%         for j = 1:q          
%             m(1,2+(j-1)*q:1+j*q) = max([zeros(1,nd);(dot(x(i,:),v(j,:))-d(j,:))/(dot(v(j,:),v(j,:)))*v(j,:)],[],1);
%         end
%         M(i,:) = m;
%     end
% 
%     y = M*beta;
%     figure(2); set(gcf,'position',[550,50,500,500]); hold on
%     box on
%     surf(X1,X2,reshape(y,prec,prec))
%     set(gca,'YTickLabel',[]);
%     set(gca,'XTickLabel',[]);
%     set(gca,'ZTickLabel',[]);
%     set(gca,'XTick',[]);
%     set(gca,'YTick',[]);
%     set(gca,'ZTick',[]);
%     
%     x = linspace(-5,5,1001);
%     plot3(x,(d(1,1) - x.*v(1,1))./v(2,1),-7*ones(1001,1),'LineWidth',1.5,'Color','black')
%     plot3(x,(d(2,1) - x.*v(1,2))./v(2,2),-7*ones(1001,1),'LineWidth',1.5,'Color','black')
%     axis([-5 5 -5 5 -7 1.5])
%     xlabel('Input 1: $x_1$','Interpreter','latex')
%     ylabel('Input 2: $x_2$','Interpreter','latex')
%     zlabel('Output: $y$','Interpreter','latex')
%     set(gca,'linewidth',1.5,'FontSize',15)
%     view(40,20); pbaspect([1, 1, 1]);
%     annotation('textbox',[.26 .01 .3 .3],'String','$\xi_1$','FitBoxToText','on','Interpreter','latex','BackgroundColor','white','FontSize',15,'EdgeColor','white','FaceAlpha',0,'LineWidth',0.01);
%     annotation('textbox',[.50 .01 .3 .3],'String','$\xi_2$','FitBoxToText','on','Interpreter','latex','BackgroundColor','white','FontSize',15,'EdgeColor','white','FaceAlpha',0,'LineWidth',0.01);
%     annotation('textbox',[.74 .01 .3 .3],'String','$\xi_3$','FitBoxToText','on','Interpreter','latex','BackgroundColor','white','FontSize',15,'EdgeColor','white','FaceAlpha',0,'LineWidth',0.01);
%     print(gcf,'JUMP_basis.png','-dpng','-r1200');

elseif Prob_ID == 6
    %%
    V = [1.5,1;1,1.8];
    D = [-3;4];
    q = 2;
    nd = 2;
    beta = [0;-0.9;1;1;2];        % beta = [0;-2;1;1;2];
    tau = [1;0;-3;0;6];
    f = @(x) griewank_JUMP_NDF(x,V,D,q,nd,beta,tau);
    l = @(x) griewank_JUMP_NDF_l(x,V,D,q,nd);
    Bayes_nMcSamples = 100; 
    nSamples = 40;
    lCheck = 0;
    while size(unique(lCheck),1) < 3
        xTrain = (lhsdesign(nSamples,2)-0.5).*10;
        lCheck = l(xTrain);
    end
    xMin = [-5;-5];   xMax = [5;5];
    lTrain = [V,D];
    yTrain = zeros(nSamples,1);
    for i = 1:nSamples
        yTrain(i,:) = f(xTrain(i,:));
    end
    DC_Type = 'Jump_NonDif';                                                       % Jump, NonDif, Jump_NonDif 
    Sigma = [0.0025,0;0,0.0005];
    DC_data = struct('f',f,'l',l,'V',V,'D',D,'DC_Type',DC_Type,...
        'Bayes_nMcSamples',Bayes_nMcSamples,'nSamples',nSamples,...
        'xTrain',xTrain,'yTrain',yTrain,'lTrain',lTrain,...
        'xMin',xMin,'xMax',xMax,'Sigma',Sigma);
    

    %%
%     figure
%     prec= 500;
%     [X1,X2] = meshgrid(linspace(-5,5,prec),linspace(-5,5,prec));
%     Y = zeros(prec,prec);
%     for i = 1:prec
%         Y(i,:) = f([X1(i,:)',X2(i,:)']);
%     end
%     surf(X1,X2,Y,'linestyle','none');    
end