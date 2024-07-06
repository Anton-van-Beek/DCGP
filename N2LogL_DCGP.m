%%

function Llh = N2LogL_DCGP(xn,xTrain,yn,d,Dn,Parm,nSamples,lvlsPhasesMat,lTrain,q,DC_Type)
omega = Parm(1,1:d);

if strcmp(DC_Type,'JUMP')
    tau = [0,Parm(1,1+d:end)];
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
    tau = [0,Parm(1,1+d:end)];
    ynCont = yn - lvlsPhasesMat*tau';
    V = lTrain(:,1:end-1);
    m = ones(1,1+q*d);
    M = ones(nSamples,1+q*d);
    D = lTrain(:,end);
    for i = 1:nSamples
        for j = 1:q
            %sign(V(j,:))*
            m(1,2+(j-1)*q:1+j*q) = max([zeros(1,d);(dot(xTrain(i,:),V(j,:))-D(j,:))/(dot(V(j,:),V(j,:)))*V(j,:)],[],1);
        end
        M(i,:) = m;
    end
end


% prec = 101;
% x = linspace(0,1,prec);
% y = linspace(0,1,prec);
% [X, Y] = meshgrid(x,y);
% 
% lTrain
% v1 = [-415/1.2, 1];
% v2 = [0.00001, 1];
% 
% value = zeros(prec,prec);
% for i = 1:prec
%     for j = 1:prec
% %         value(i,j) = [X(i,j),Y(i,j)]*v - d;
%         if [X(i,j),Y(i,j)]*v1' - Dn(1,:) > 0
%             value(i,j) = -1;
%         end
% 
% 
%         if [X(i,j),Y(i,j)]*v2' - Dn(2,:) > 0
%             value(i,j) = 1;
%         end
% 
% 
%     end
% end
% surf(X,Y,value)
% fdgdfg

R = corrmat_DCGP(xn,xn,omega);

R = (R + R')/2;
EigMin = min(eig(R));
if EigMin < 1e-8 
    Nug = 1e-8 - EigMin;
    R = R + eye(nSamples)*Nug;
end

Riy = R\ynCont;
RiM = R\M;
beta = (M'*RiM)\M'*Riy;                         %Prior weights
Riymb = R\(ynCont - M*beta);

s2 = (1/nSamples)*(ynCont - M*beta)'*Riymb;     %Prior variance


Llh = nSamples*log(s2) + log(det(R));

end