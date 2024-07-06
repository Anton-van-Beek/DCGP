function model = DCGP_fit(xTrain,yTrain,lTrain,V,D,varargin)

%% Check the inputs
InParse = inputParser;
InParse.CaseSensitive = 0;
InParse.KeepUnmatched = 0;
InParse.PartialMatching = 1;
InParse.StructExpand = 1;

vf1 = @(x) isnumeric(x) && isreal(x);
addRequired(InParse,'xTrain', vf1);
addRequired(InParse,'yTrain', vf1);
addRequired(InParse,'lTrain', vf1);
addOptional(InParse, 'DC_Type', 'Jump', @(x) any(validatestring(x, {'Jump', 'NonDif', 'Jump_NonDif'})));
addOptional(InParse, 'l', 1);

parse(InParse,xTrain,yTrain,lTrain,varargin{:});
DC_Type = upper(InParse.Results.DC_Type);
l = InParse.Results.l;

%Set training parameters
options = optimoptions('fmincon','Display', 'iter-detailed','TolCon',1e-6,'TolFun',1e-12, 'MaxIter', 2000, 'MaxFunEvals', 50000, ...
                'CheckGradients', false, 'FiniteDifferenceType', 'forward', 'ScaleProblem', 'obj-and-constr', ...
				'Algorithm', 'sqp');
oMin =  1;              oMax = 1.335;
tMin = -2;              tMax = 2;

nSamples = size(xTrain,1);
nLlhSamples = 25;
lvlsPhases = unique(lTrain);
nPhase = size(lvlsPhases,1);
xMin = min(xTrain);
xMax = max(xTrain);
yMin = min(yTrain);
yMax = max(yTrain);
d = size(xTrain,2);
q = size(lTrain,1);

%% Normalize the training data
xn = (xTrain - xMin)./(xMax - xMin);
yn = (yTrain - yMin)./(yMax - yMin);

%% Initialize and run the optimization protocol

if strcmp(DC_Type,'JUMP')
    lvlsPhasesMat = zeros(nSamples,nPhase);
    for i = 0:nPhase-1
        for j = 1:nSamples
            if lTrain(j,1) == i
                lvlsPhasesMat(j,i+1) = 1;
            end
        end
    end
    Oini = lhsdesign(nLlhSamples,d).*repmat(oMax - oMin,nLlhSamples,d) + repmat(oMin,nLlhSamples,d);
    Pini = lhsdesign(nLlhSamples,nPhase-1).*repmat(tMax - tMin,nLlhSamples,nPhase-1) + repmat(tMin,nLlhSamples,nPhase-1);
    HPini = [Oini,Pini];
    LowerBound = [repmat(oMin,d,1);repmat(tMin,nPhase-1,1)];
    UpperBound = [repmat(oMax,d,1);repmat(tMax,nPhase-1,1)];
    
    ParmStorage = zeros(nLlhSamples,d+nPhase-1);
    LlhStorage = zeros(nLlhSamples,1);
    Dn = 0;
    for i = 1:nLlhSamples
        [ParmStorage(i,:),LlhStorage(i,1)] = fmincon(@(Parm) N2LogL_DCGP(xn,xTrain,yn,d,Dn,Parm,nSamples,lvlsPhasesMat,lTrain,q,DC_Type), HPini(i,:), [], [], [], [], LowerBound, UpperBound, [], options);
    end
    [~, ind] = min(LlhStorage);
    omega = ParmStorage(ind,1:d);
    tau = [0,ParmStorage(ind,1+d:end)];
elseif strcmp(DC_Type,'NONDIF')
    lvlsPhasesMat = 1;
    Oini = lhsdesign(nLlhSamples,d).*repmat(oMax - oMin,nLlhSamples,d) + repmat(oMin,nLlhSamples,d);
    HPini = Oini;
    LowerBound = repmat(oMin,d,1);
    UpperBound = repmat(oMax,d,1);
    Dn =  (lTrain(:,end) - repmat(sum(xMin),q,1))./repmat(norm(xMax - xMin),q,1);

    ParmStorage = zeros(nLlhSamples,d);
    LlhStorage = zeros(nLlhSamples,1);
    for i = 1:nLlhSamples
        try
            [ParmStorage(i,:),LlhStorage(i,1)] = fmincon(@(Parm) N2LogL_DCGP(xn,xTrain,yn,d,Dn,Parm,nSamples,lvlsPhasesMat,lTrain,q,DC_Type), HPini(i,:), [], [], [], [], LowerBound, UpperBound, [], options);
        catch
            disp(num2str(i))
        end
    end
    [~, ind] = min(LlhStorage);
    omega = ParmStorage(ind,1:d);
    tau = 0;
elseif strcmp(DC_Type,'JUMP_NONDIF')
    lvlsPhasesVec = l(xTrain);
    nPhase = size(unique(lvlsPhasesVec),1);
    lvlsPhasesMat = zeros(nSamples,nPhase);
    for i = 0:nPhase-1
        for j = 1:nSamples
            if lvlsPhasesVec(j,1) == i
                lvlsPhasesMat(j,i+1) = 1;
            end
        end
    end    

    Oini = lhsdesign(nLlhSamples,d).*repmat(oMax - oMin,nLlhSamples,d) + repmat(oMin,nLlhSamples,d);
    Pini = lhsdesign(nLlhSamples,nPhase-1).*repmat(tMax - tMin,nLlhSamples,nPhase-1) + repmat(tMin,nLlhSamples,nPhase-1);
    HPini = [Oini,Pini];
    LowerBound = [repmat(oMin,d,1);repmat(tMin,nPhase-1,1)];
    UpperBound = [repmat(oMax,d,1);repmat(tMax,nPhase-1,1)];

    Dn =  (lTrain(:,end) - repmat(sum(xMin),q,1))./repmat(norm(xMax - xMin),q,1);
    ParmStorage = zeros(nLlhSamples,d+nPhase-1);
    LlhStorage = zeros(nLlhSamples,1);
    for i = 1:nLlhSamples
        try
            [ParmStorage(i,:),LlhStorage(i,1)] = fmincon(@(Parm) N2LogL_DCGP(xn,xTrain,yn,d,Dn,Parm,nSamples,lvlsPhasesMat,lTrain,q,DC_Type), HPini(i,:), [], [], [], [], LowerBound, UpperBound, [], options);
        catch

        end
    end
    [~, ind] = min(LlhStorage);
    omega = ParmStorage(ind,1:d);
    tau = [0,ParmStorage(ind,1+d:end)];
end

%% Plot loglikelihood
% figure; set(gcf,'position',[50,50,600,500]); hold on
% nPrec = 100;
% llhsurf = zeros(nPrec,nPrec);
% [X,Y] = meshgrid(linspace(oMin,oMax,nPrec),linspace(tMin,tMax,nPrec));
% for i = 1:nPrec
%     for j = 1:nPrec
%         llhsurf(i,j) = N2LogL_DCGP(xn,yn,d,Dn,[X(i,j),Y(i,j)],nSamples,lvlsPhasesMat,lTrain,q,DC_Type);
%     end
% end
% %h = surf(linspace(oMin,oMax,nPrec),linspace(tMin,tMax,nPrec),exp(-llhsurf-q));
% h = surf(linspace(oMin,oMax,nPrec),linspace(tMin,tMax,nPrec),llhsurf);
% set(h,'edgecolor','none');
% ylabel('$\tau_1$','Interpreter','latex')
% xlabel('$\omega$','Interpreter','latex')
% view(0,90);
% title('LLH');
% box on 
% set(gca,'linewidth',1.5,'FontSize',15)
%axis([oMin,oMax,tMin,tMax,min(min(llhsurf))-1,max(max(llhsurf))+1])

%% Post process the training results to facilitate 
R = corrmat_DCGP(xn,xn,omega);
R = (R + R')/2;

if strcmp(DC_Type,'JUMP')
    ynCont = yn - lvlsPhasesMat*tau';
    M = ones(nSamples,1);
elseif strcmp(DC_Type,'NONDIF')
    ynCont = yn;
    V = lTrain(:,1:end-1);
    m = ones(1,1+q*d);
    M = ones(nSamples,1+q*d);
    D = lTrain(:,end);
    for i = 1:nSamples
        for j = 1:q
            m(1,2+(j-1)*q:1+j*q) = max([zeros(1,d);(dot(xTrain(i,:),V(j,:))-D(j,:))/(dot(V(j,:),V(j,:)))*V(j,:)],[],1);
        end
        M(i,:) = m;
    end
elseif strcmp(DC_Type,'JUMP_NONDIF')
    ynCont = yn - lvlsPhasesMat*tau';
    V = lTrain(:,1:end-1);
    m = ones(1,1+q*d);
    M = ones(nSamples,1+q*d);
    D = lTrain(:,end);
    for i = 1:nSamples
        for j = 1:q
            m(1,2+(j-1)*q:1+j*q) = max([zeros(1,d);(dot(xTrain(i,:),V(j,:))-D(j,:))/(dot(V(j,:),V(j,:)))*V(j,:)],[],1);
        end
        M(i,:) = m;
    end
end

EigMin = min(eig(R));
if EigMin < 1e-8 
    Nug = 1e-8 - EigMin;
    R = R + eye(nSamples)*Nug;
else
    Nug = 0;
end
Riy = R\ynCont;
RiM = R\M;
beta = (M'*RiM)\M'*Riy;                         %Prior weights
Riymb = R\(ynCont - M*beta);
s2 = (1/nSamples)*(ynCont - M*beta)'*Riymb;     %Prior variance

%% Save the data to a structure 
model = struct('xn',xn,'xTrain',xTrain,'yn',yn,'yTrain',yTrain,'ynCont',ynCont,'omega',omega,'tau',tau,'R',R,'M',M,...
    'beta',beta,'s2',s2,'n',nSamples,'Nug',Nug,'xMin',xMin,'Dn',Dn,'l',l,...
    'Riymb',Riymb,'xMax',xMax,'yMin',yMin,'DC_Type',DC_Type,'LB',LowerBound,'UB',UpperBound,...
    'yMax',yMax,'d',d,'q',q,'nPhase', nPhase,'lvlsPhases',lvlsPhases,'lTrain',lTrain);
end