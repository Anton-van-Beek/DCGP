function [yPred,yStd] = Pred_DCGP(xPred,lPred,nPlot,model,varargin)
    InParse = inputParser;
    InParse.CaseSensitive = 0;
    InParse.KeepUnmatched = 0;
    InParse.PartialMatching = 1;
    InParse.StructExpand = 1;
    
    vf1 = @(x) isnumeric(x) && isreal(x);
    addRequired(InParse,'xPred', vf1);
    addRequired(InParse,'lPred', vf1);
    addRequired(InParse,'nPlot', vf1);
    addRequired(InParse,'model', @isstruct);

    addOptional(InParse, 'Bayes_Type', 'Partial', @(x) any(validatestring(x, {'Full', 'Partial'})));
    addOptional(InParse, 'Bayes_nMcSamples', 1000, @(x) isnumeric(x) && x>=1);  

    parse(InParse,xPred,lPred,nPlot,model,varargin{:});
    Bayes_Type = upper(InParse.Results.Bayes_Type);
    Bayes_nMcSamples = upper(InParse.Results.Bayes_nMcSamples);
    xnPred = (xPred - repmat(model.xMin,nPlot,1) )./(repmat(model.xMax - model.xMin,nPlot,1));   
    
    if strcmp(Bayes_Type,'PARTIAL')
        if strcmp(model.DC_Type,'JUMP')
            M = ones(1,nPlot);
            lvlsPhasesVec = model.l(xPred);
            lvlsPhasesMatPred = zeros(nPlot,model.nPhase);
            for i = 0:model.nPhase-1
                for j = 1:nPlot
                    if lvlsPhasesVec(j,1) == i
                        lvlsPhasesMatPred(j,i+1) = 1;
                    end
                end 
            end   
        elseif strcmp(model.DC_Type,'NONDIF')
            V = model.lTrain(:,1:end-1);
            m = ones(1,1+model.q*model.d);
            M = ones(1+model.q*model.d,nPlot);
            for i = 1:nPlot
                for j = 1:model.q
                    m(1,2+(j-1)*model.q:1+j*model.q) = max([zeros(1,model.d);(dot(xPred(i,:),V(j,:))-model.lTrain(j,end))/(dot(V(j,:),V(j,:)))*V(j,:)],[],1);
                end
                M(:,i) = m;
            end
        elseif strcmp(model.DC_Type,'JUMP_NONDIF')
            V = model.lTrain(:,1:end-1);
            m = ones(1,1+model.q*model.d);
            M = ones(1+model.q*model.d,nPlot);
            for i = 1:nPlot
                for j = 1:model.q
                    m(1,2+(j-1)*model.q:1+j*model.q) = max([zeros(1,model.d);(dot(xPred(i,:),V(j,:))-model.lTrain(j,end))/(dot(V(j,:),V(j,:)))*V(j,:)],[],1);
                end
                M(:,i) = m;
            end
            lvlsPhasesVec = model.l(xPred);
            lvlsPhasesMatPred = zeros(nPlot,model.nPhase);
            for i = 0:model.nPhase-1
                for j = 1:nPlot
                    if lvlsPhasesVec(j,1) == i
                        lvlsPhasesMatPred(j,i+1) = 1;
                    end
                end
            end    
        end

        r = corrmat_DCGP(model.xn,xnPred,model.omega);
        ynPred = M'*model.beta + r'*model.Riymb;
        W = M - model.M'*(model.R\r);
        ynMse = model.s2*(1 - r'*(model.R\r) + W'*((model.M'*(model.R\model.M))\W));
        if max(strcmp(model.DC_Type,{'JUMP','JUMP_NONDIF'}))
            yPred = (ynPred + lvlsPhasesMatPred*model.tau').*(model.yMax-model.yMin) + model.yMin;
        elseif strcmp(model.DC_Type,'NONDIF')
            yPred = (ynPred).*(model.yMax-model.yMin) + model.yMin;
        end
        yStd = sqrt(diag(ynMse)).*repmat(model.yMax-model.yMin,nPlot,1);

        %% Bayes prediction GP
    elseif strcmp(Bayes_Type,'FULL')
        % First, we get the probability of a MC sample of hyperparameters
        dHypParm = size(model.LB,1);
        McSamples = lhsdesign(Bayes_nMcSamples,dHypParm)'.*repmat(model.UB-model.LB,1,Bayes_nMcSamples)+repmat(model.LB,1,Bayes_nMcSamples);

        PropVec = zeros(1,Bayes_nMcSamples);
        McSamMat = zeros(nPlot,Bayes_nMcSamples);
        nPlot = nPlot+model.n;
        xn_xnPred = [model.xn;xnPred];
        x_xPred = [model.xTrain;xPred];

        if strcmp(model.DC_Type,'JUMP')
            lPred = [model.lTrain;lPred];
            lvlsPhasesMat = zeros(model.n,model.nPhase);
            for i = 0:model.nPhase-1
                for j = 1:model.n
                    if model.lTrain(j,1) == i
                        lvlsPhasesMat(j,i+1) = 1;
                    end
                end
            end
        elseif strcmp(model.DC_Type,'NONDIF')
            lvlsPhasesMat= 1; 
        elseif strcmp(model.DC_Type,'JUMP_NONDIF')
            lvlsPhasesVec = model.l(model.xTrain);
            lPred = [model.l(model.xTrain);lPred];
            lvlsPhasesMat = zeros(model.n,model.nPhase);
            for i = 0:model.nPhase-1
                for j = 1:model.n
                    if lvlsPhasesVec(j,1) == i
                        lvlsPhasesMat(j,i+1) = 1;
                    end
                end
            end
        end

        for k = 1:Bayes_nMcSamples
            if strcmp(model.DC_Type,'JUMP')
                lvlsPhasesMatPred = zeros(model.q,model.nPhase);
                for i = 0:model.nPhase-1
                    for j = 1:model.q
                        if model.lTrain(j,1) == i
                            lvlsPhasesMatPred(j,i+1) = 1;
                        end
                    end
                end
                tau = [0,McSamples(1+model.d:end,k)'];
                ynCont = model.yn - lvlsPhasesMatPred*tau';
                M = ones(model.q,1);
            elseif strcmp(model.DC_Type,'NONDIF')
                ynCont = model.yn;
                V = model.lTrain(:,1:end-1);
                m = ones(1,1+model.q*model.d);
                M = ones(model.n,1+model.q*model.d);
                for i = 1:model.n
                    for j = 1:model.q
                        m(1,2+(j-1)*model.q:1+j*model.q) = max([zeros(1,model.d);(dot(model.xTrain(i,:),V(j,:))-model.lTrain(j,end))/(dot(V(j,:),V(j,:)))*V(j,:)],[],1);
                    end 
                    M(i,:) = m;
                end
            elseif strcmp(model.DC_Type,'JUMP_NONDIF')
                lvlsPhasesMatPred = zeros(model.n,model.nPhase);
                for i = 0:model.nPhase-1
                    for j = 1:model.n
                        if lvlsPhasesVec(j,1) == i
                            lvlsPhasesMatPred(j,i+1) = 1;
                        end
                    end
                end
                tau = [0,McSamples(1+model.d:end,k)'];
                ynCont = model.yn - lvlsPhasesMatPred*tau';

                V = model.lTrain(:,1:end-1);
                m = ones(1,1+model.q*model.d);
                M = ones(model.n,1+model.q*model.d);
                for i = 1:model.n
                    for j = 1:model.q
                        m(1,2+(j-1)*model.q:1+j*model.q) = max([zeros(1,model.d);(dot(model.xTrain(i,:),V(j,:))-model.lTrain(j,end))/(dot(V(j,:),V(j,:)))*V(j,:)],[],1);
                    end
                    M(i,:) = m;
                end
            end
            R = corrmat_DCGP(model.xn,model.xn,McSamples(1:model.d,k)');
            R = (R + R')/2;
            EigMin = min(eig(R));
            if EigMin < 1e-8 
                Nug = 1e-8 - EigMin;
                R = R + eye(model.n)*Nug;
            end
            Riy = R\ynCont;
            RiM = R\M;
            beta = (M'*RiM)\M'*Riy;                        %Prior weights

            Riymb = R\(ynCont - M*beta);
            s2 = (1/model.q)*(ynCont - M*beta)'*Riymb;     %Prior variance
            PropVec(1,k) = exp(-N2LogL_DCGP(model.xn,model.xTrain,model.yn,model.d,model.Dn,McSamples(:,k)',model.n,lvlsPhasesMat,model.lTrain,model.q,model.DC_Type)-model.n);         
            % Second, we generate a rnandom samples of responses xPred

            if strcmp(model.DC_Type,'JUMP')
                Mpred = ones(1,nPlot);
                lvlsPhasesMatPred = zeros(nPlot,model.nPhase);
                for i = 0:model.nPhase-1
                    for j = 1:nPlot
                        if lPred(j,1) == i
                            lvlsPhasesMatPred(j,i+1) = 1;
                        end
                    end
                end
            elseif strcmp(model.DC_Type,'NONDIF')
                V = model.lTrain(:,1:end-1);
                m = ones(1,1+model.q*model.d);
                Mpred = ones(1+model.q*model.d,nPlot);
                for i = 1:nPlot
                    for j = 1:model.q
                        m(1,2+(j-1)*model.q:1+j*model.q) = max([zeros(1,model.d);(dot(x_xPred(i,:),V(j,:))-model.lTrain(j,end))/(dot(V(j,:),V(j,:)))*V(j,:)],[],1);
                    end
                    Mpred(:,i) = m; 
                end
            elseif strcmp(model.DC_Type,'JUMP_NONDIF')
                lvlsPhasesMatPred = zeros(nPlot,model.nPhase);
                for i = 0:model.nPhase-1
                    for j = 1:nPlot
                        if lPred(j,1) == i
                            lvlsPhasesMatPred(j,i+1) = 1;
                        end
                    end
                end
                V = model.lTrain(:,1:end-1);
                m = ones(1,1+model.q*model.d);
                Mpred = ones(1+model.q*model.d,nPlot);
                for i = 1:nPlot
                    for j = 1:model.q
                        m(1,2+(j-1)*model.q:1+j*model.q) = max([zeros(1,model.d);(dot(x_xPred(i,:),V(j,:))-model.lTrain(j,end))/(dot(V(j,:),V(j,:)))*V(j,:)],[],1);
                    end
                    Mpred(:,i) = m;
                end
            end

            r = corrmat_DCGP(model.xn,xn_xnPred,McSamples(1:model.d,k)');  
            ynPred = Mpred'*beta + r'*Riymb;
            W = (Mpred' - (model.M'*(R\r))')';
            ynMse = s2*(1 - r'*(R\r) + W'*((model.M'*(R\model.M))\W) + eye(nPlot)*model.Nug);
            if strcmp(model.DC_Type,'JUMP')
                Scale = mean(ynPred(1:model.q,1) + lvlsPhasesMatPred(1:model.q,:)*tau' - model.yn);
                yPred = (ynPred(1+model.n:end,1) + lvlsPhasesMatPred(1+model.n:end,:)*tau' - Scale).*(model.yMax-model.yMin) + model.yMin;
                yMse = ynMse.*(model.yMax-model.yMin);
            elseif strcmp(model.DC_Type,'NONDIF')
                yMse = ynMse.*(model.yMax-model.yMin);
                yPred = (ynPred(1+model.n:end,1)).*(model.yMax-model.yMin) + model.yMin;
            elseif strcmp(model.DC_Type,'JUMP_NONDIF')
                yPred = (ynPred(1+model.n:end,1) + lvlsPhasesMatPred(1+model.n:end,:)*tau').*(model.yMax-model.yMin) + model.yMin;
                yMse = ynMse.*(model.yMax-model.yMin);
            end 
            yMse = diag(yMse);
            yMse(yMse<0) = 0;

            McSamMat(:,k) = yPred + mvnrnd( zeros(1,nPlot-model.n), yMse(1+model.n:end,1)' )';
        end
        PropVec(isnan(PropVec)) = 0;
        weights = PropVec./sum(PropVec);
        NanIndex = isnan(McSamMat);
        weights(:,NanIndex(1,:)) = [];
        McSamMat(:,NanIndex(1,:)) = [];
        yPred = (weights*McSamMat')';
        yStd = (sqrt(weights*(McSamMat.^2)' - (yPred').^2))';
    end

    
end