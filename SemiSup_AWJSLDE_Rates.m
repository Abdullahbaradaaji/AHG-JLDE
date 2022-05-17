function[NL_Rate,L_Rate,Test_Rate]=SemiSup_AWJSLDE_Rates(t_Alpha,t_Beta,t_Gamma,knn,DB_name,nTrnSmp,NSplits,tableof_k,PCA_cond,PCA_Ratio)
%Inputs
%             t_Alpfa = Table of Alpha Value , t_Beta = Table of Beta Value
%             , t_Gamma = Table of Gamma Value
%             DB_name : name of database in string format
%             nTrnSmp : number of train sample for each class
%             NSplits : number of random splits
%             tableof_k = table of number of labeled data per class
%             type= PCA or NO_PCA

%Output
%             NL_Unlabeled_Rate = non lineair rate of unlabeled train
%             L_Unlabeled_Rate =  lineair rate of unlabeled train
%             Test_Rate = rate of test data

tic

%Call splitDBs function
data = splitDBs(DB_name , nTrnSmp , NSplits);
if strcmp (PCA_cond ,'PCA')
    for i=1:NSplits
        [data{i,1}.fea,data{i,2}.fea] = PCA(data{i,1}.fea, data{i,2}.fea, PCA_Ratio);
    end
else
end


%Call ArrangedData function
NL_Rate = [];
L_Rate = [];
Test_Rate = [];
for k=1:length(tableof_k)
    NLabeled_class = tableof_k(k);
    DATA= ArrangeData(data,NSplits,NLabeled_class);

    counter=0;

    
    for a=1:length (t_Alpha)
        Alpha = t_Alpha (a);
        for b=1:length (t_Beta)
            Beta = t_Beta (b);
            for g=1:length (t_Gamma)
                Gamma = t_Gamma (g);
                counter = counter +1;
                %                 L_Unlabeled_ALL=[];
                %                 L_Test_ALL=[];
                for i = 1 : NSplits
                    
                    %Calculate F and A
                    
                    [A,F]= SemiSupervised_AWJSLDE(DATA{i,1},Alpha, Beta , Gamma,knn );
                    
                    % non lineair Rate of calculated class membership (comparing F with labels )
                    [~, index] = max(F,[],2);
                    index = index (size(DATA{i,1}.labeledgnd,1)+1 :size (F,1) );
                    score=0;
                    
                    for j=1 : size(index,1)
                        %score = score + index (j)== DATA{i,1}.unlabeledgnd(j);
                        if index (j)== DATA{i,1}.unlabeledgnd(j)
                            score = score +1;
                        end
                    end
                    
                    
                    NL_Unlabeled_Rate(i) = 100*score/size(index,1);
                    
                    %  lineair Rate
                    P_UnLabeled_Data = A'*DATA{i,1}.unlabeledfea;% projection of UnLabeled Training Data
                    P_Labeled_Data = A'*DATA{i,1}.labeledfea;% projection of Labeled Training Data
                    P_Testing_Data = A'*DATA{i,2}.fea;% projection of testing  Data
                    for d=1:size(A,2)
                        P_UnLabeled_Data_DIMVARI =  P_UnLabeled_Data (1:d,:); %  DIMVARI = variable Dimension
                        P_Labeled_Data_DIMVARI = P_Labeled_Data (1:d,:);
                        P_Testing_Data_DIMVARI = P_Testing_Data (1:d,:);
                        
                        
                       
                        class =  fitcknn( P_Labeled_Data_DIMVARI', DATA{i,1}.labeledgnd);
                        
                        
                        L_Unlabeled(d) = sum((predict(class,P_UnLabeled_Data_DIMVARI') == DATA{i,1}.unlabeledgnd))/length(DATA{i,1}.unlabeledgnd)*100;
                        
                        L_Test(d) = sum((predict(class,P_Testing_Data_DIMVARI') == DATA{i,2}.gnd))/length(DATA{i,2}.gnd)*100;
                        
                    end
                    
                    eval(['L_Rate.',DB_name,'.Split' , int2str(i), '(' , int2str(counter), ',k)=max(L_Unlabeled);']);
                    eval(['Test_Rate.',DB_name,'.Split' , int2str(i), '(' , int2str(counter), ',k)=max(L_Test);']);
                    eval(['NL_Rate.',DB_name,'.Split' , int2str(i), '(' , int2str(counter), ',k)=max(NL_Unlabeled_Rate(i));']);
                    
                end
                
                %  calculation of Non lineair Mean  Rate
                %                 NL_Unlabeled_Rate = sum (NL_Unlabeled_Rate) / NSplits;
                %
                %                 L_Unlabeled_MeanRate = sum (L_Unlabeled) / NSplits;
                %
                %                 L_Test_MeanRate = sum (L_Test) / NSplits;
                
                %NL_Rate = [NL_Rate ; NL_Unlabeled_Rate ];
                %L_Rate = [L_Rate; L_Unlabeled_MeanRate];
                %Test_Rate = [Test_Rate;L_Test_MeanRate];
                %save([DB_name 'SemiSupervisedRateAlgo.mat'],'L_Rate','Test_Rate','NL_Rate');
                disp([num2str(counter), '/', num2str(length(t_Alpha)*length(t_Beta)*length(t_Gamma)), ' Completed']);
                toc
                 clear A F  L_Test L_Unlabeled;
            end
        end
    end
    %             end
    %         end
    %     end
    
    
    eval(['sumL_Rate = zeros(size(L_Rate.', DB_name,'.Split',int2str(NSplits),'));']);
    eval(['sumTest_Rate = zeros(size(Test_Rate.', DB_name,'.Split',int2str(NSplits),'));']);
    eval(['sumNL_Rate = zeros(size(NL_Rate.', DB_name,'.Split',int2str(NSplits),'));']);
    for i = 1 : NSplits
        
        eval(['sumL_Rate = sumL_Rate + L_Rate.', DB_name, '.Split', int2str(i), ';']);
        eval(['sumTest_Rate = sumTest_Rate + Test_Rate.', DB_name, '.Split', int2str(i), ';']);
        eval(['sumNL_Rate = sumNL_Rate + NL_Rate.', DB_name, '.Split', int2str(i), ';']);
    end
    
    eval(['L_Rate.',DB_name,'.MeanSplits = sumL_Rate ./ NSplits;']);
    eval(['Test_Rate.',DB_name,'.MeanSplits = sumTest_Rate ./ NSplits;']);
    eval(['NL_Rate.',DB_name,'.MeanSplits = sumNL_Rate ./ NSplits;']);
end
for i=1:length(tableof_k)
    eval(['L_Rate.',DB_name,'.MaxOfMeanSplits(i) = max(L_Rate.',DB_name,'.MeanSplits(:,i));']);
    eval(['Test_Rate.',DB_name,'.MaxOfMeanSplits(i) = max(Test_Rate.',DB_name,'.MeanSplits(:,i));']);
    eval(['NL_Rate.',DB_name,'.MaxOfMeanSplits(i) = max(NL_Rate.',DB_name,'.MeanSplits(:,i));']);
end



end
