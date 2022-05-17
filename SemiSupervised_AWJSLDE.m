

function [A,F]= SemiSupervised_AWJSLDE(X ,Alpha, Beta , Gamma,knn )
% X is a structure contain 6 fields labeled data d*m, labels of labeled data m*1 , unlabeled data d*n , labels on the unlabeled data n*1 , arranged matrix of labeled and unlabeled data d*N where N = m + n, ana arranged matrix of the labels N*1
%  Alpha , Beta, and gamma are balanced parameters example (10^6)
%knn is an interger knn = 7 or 10

%output
%         A is the linear transform matrix
%         F is the predicted label matrix


%% Estimation of A
%% Calculate F from FME
C = max(X.arrangedgnd);
Y=zeros (length(X.arrangedgnd),C);% Fl : gnd of labeled data
for i=1:length(X.arrangedgnd)
    Y(i,X.arrangedgnd(i))=1;
end
%compute data graph
[~,W]=KNN_GraphConstruction(X.arrangedfea,3);
L1=diag(sum(W))-W;

F=ones (length(X.arrangedgnd),C);
F=F/C;
for i=1:length(X.labeledgnd)
    F(i,X.labeledgnd(i))=1;
end


Fu=F(length(X.labeledgnd)+1:end ,:);

F=[Y(1:size(X.labeledgnd,1),:);Fu];


Fs = zscore(F');
Fs=Fs';
S2=Fs*Fs';
S2=abs(S2);
Fu=Fu/sum(sum(Fu));

F=[Y(1:size(X.labeledgnd,1),:);Fu];

[val,idx]=sort(S2,'descend');

%compute label graph
S2_graph = zeros(size(S2));

for i=1:knn
    for  j=1:length(S2_graph)
        S2_graph(idx(i,j),j)=val(i,j);
    end
end
for i=1:length(S2_graph)
    for  j=1:length(S2_graph)
        S2_graph(i,j)=S2_graph(j,i);
    end
end

L2=diag(sum(S2_graph))-S2_graph;


%%%%creation of U %%%%%%%%%%%%%%%%%%%%%%
U=zeros(size(X.arrangedfea,2));
for i=1:size(X.labeledfea,2)
    U(i,i)=1E2;
end



%Compute A linear transform matrix 
Wb=ones(size(F,1))- (F*F');
Db=diag(sum(Wb));


w1=1;v1=1;w2=1;v2=1; % initial weighted values

lambda = 1e-6;
[A, eigVal] = eig((X.arrangedfea*Db*X.arrangedfea'+lambda*eye(size(X.arrangedfea,1)))\(X.arrangedfea*Wb*X.arrangedfea'+(Gamma*(v1*(X.arrangedfea*L1*X.arrangedfea') + v2*(X.arrangedfea*L2*X.arrangedfea')))));

[~, idx ] = sort(real(diag(eigVal)) ,'ascend');
A = real(A(:,idx));

%Compute F
Z = X.arrangedfea'*(A*A')*X.arrangedfea;
Z_positive=(Z+abs(Z))/2;
Z_negative=(abs(Z)-Z)/2;

L_final = w1*L1 + w2*L2;
L_final_positive=(L_final+abs(L_final))/2;
L_final_negative=(abs(L_final)-L_final)/2;

temp_Neg=(Beta*L_final_positive*F)+(Z_negative*F)+(U*F)+(2*Alpha*F*(F'*F));
temp_Pos=(Beta*L_final_negative*F)+(Z_positive*F)+(U*Y)+(2*Alpha*F);

for i=1:length(X.arrangedgnd)
    for j=1:C
        F(i,j)=(temp_Pos(i,j)/temp_Neg(i,j))*F(i,j);
    end
end


Fu=F(length(X.labeledgnd)+1:end ,:);

F=[Y(1:size(X.labeledgnd,1),:);Fu];



Fs = zscore(F');
Fs=Fs';
S2=Fs*Fs';
S2=abs(S2);
Fu=Fu/sum(sum(Fu));

F=[Y(1:size(X.labeledgnd,1),:);Fu];

[val,idx]=sort(S2,'descend');

S2_graph = zeros(size(S2));

for i=1:knn
    for  j=1:length(S2_graph)
        S2_graph(idx(i,j),j)=val(i,j);
    end
end
for i=1:length(S2_graph)
    for  j=1:length(S2_graph)
        S2_graph(i,j)=S2_graph(j,i);
    end
end

L2=diag(sum(S2_graph))-S2_graph;

f(2) = trace(A'*(X.arrangedfea*Wb*X.arrangedfea')*A)+ trace((F-Y)'*U*(F-Y))+ Beta*(w1*trace(F'*L1*F) + w2*trace(F'*L2*F))+ Gamma*(v1*trace(A'*(X.arrangedfea*L1*X.arrangedfea')*A)   +  v2*trace(A'*(X.arrangedfea*L2*X.arrangedfea')*A) )+ Alpha*trace((F'*F-eye(size(F'*F)))'*(F'*F-eye(size(F'*F))));

N=2;


while  abs(f(N)-f(N-1)) > 1e-3
    
    [~,ind]=max(F,[],2);
    
    Wb=ones(size(F,1))- (F*F');

    Db=diag(sum(Wb));
    

    w1=1/ ( 2* (trace(F'*L1*F))^0.5 );
    v1=1/ ( 2* (trace(A'*(X.arrangedfea*L1*X.arrangedfea')*A))^0.5) ;
    w2=1/ ( 2* (trace(F'*L2*F))^0.5);
    v2=1/ ( 2* (trace(A'*(X.arrangedfea*L2*X.arrangedfea')*A))^0.5) ;
    
    [A, eigVal] = eig((X.arrangedfea*Db*X.arrangedfea'+lambda*eye(size(X.arrangedfea,1)))\(X.arrangedfea*Wb*X.arrangedfea'+(Gamma*(v1*(X.arrangedfea*L1*X.arrangedfea') + v2*(X.arrangedfea*L2*X.arrangedfea')))));
    
    [~, idx ] = sort(real(diag(eigVal)) ,'ascend');
    A = real(A(:,idx));
    
    %Compute F
    Z = X.arrangedfea'*(A*A')*X.arrangedfea;
    Z_positive=[Z+abs(Z)]/2;
    Z_negative=[abs(Z)-Z]/2;
    
    L_final = w1*L1 + w2*L2;
    L_final_positive=(L_final+abs(L_final))/2;
    L_final_negative=(abs(L_final)-L_final)/2;
    
    temp_Neg=(Beta*L_final_positive*F)+(Z_negative*F)+(U*F)+(2*Alpha*F*(F'*F));
    temp_Pos=(Beta*L_final_negative*F)+(Z_positive*F)+(U*Y)+(2*Alpha*F);
    
    for i=1:length(X.arrangedgnd)
        for j=1:C
            F(i,j)=(temp_Pos(i,j)/temp_Neg(i,j))*F(i,j);
        end
    end
    
    %       F=F/sum(sum(F));
    
    
    Fu=F(length(X.labeledgnd)+1:end ,:);

    
    F=[Y(1:size(X.labeledgnd,1),:);Fu];

    Fs = zscore(F');
    Fs=Fs';
    S2=Fs*Fs';
    S2=abs(S2);
    Fu=Fu/sum(sum(Fu));

    F=[Y(1:size(X.labeledgnd,1),:);Fu];
    [val,idx]=sort(S2,'descend');
    
    S2_graph = zeros(size(S2));
    
    for i=1:knn
        for  j=1:length(S2_graph)
            S2_graph(idx(i,j),j)=val(i,j);
        end
    end
    for i=1:length(S2_graph)
        for  j=1:length(S2_graph)
            S2_graph(i,j)=S2_graph(j,i);
        end
    end
    
    L2=diag(sum(S2_graph))-S2_graph;
    
    N=N+1;
    f(N) = trace(A'*(X.arrangedfea*Wb*X.arrangedfea')*A)+trace((F-Y)'*U*(F-Y))+Beta*(w1*trace(F'*L1*F) + w2*trace(F'*L2*F))+Gamma*(v1*trace(A'*(X.arrangedfea*L1*X.arrangedfea')*A)   +  v2*trace(A'*(X.arrangedfea*L2*X.arrangedfea')*A) )+Alpha*trace((F'*F-eye(size(F'*F)))'*(F'*F-eye(size(F'*F))));
    Result = f(N)-f(N-1);
    
    if N==30
        break
        
    end
    
end
[~,ind]=max(F,[],2);
F_discrete = zeros(size(F));
for i=1:size(F,1)
    F_discrete(i,ind(i))=1;
end
Wb=ones(size(F_discrete,1))- (F_discrete*F_discrete');
Db=diag(sum(Wb));

[A, eigVal] = eig((X.arrangedfea*Db*X.arrangedfea'+lambda*eye(size(X.arrangedfea,1)))\(X.arrangedfea*Wb*X.arrangedfea'+(Gamma*(X.arrangedfea*L_final*X.arrangedfea'))));

[~, idx ] = sort(real(diag(eigVal)) ,'ascend');
A = real(A(:,idx));

F=F/sum(sum(F));


end
