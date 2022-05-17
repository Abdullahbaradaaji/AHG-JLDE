function labeledunlabeledData= ArrangeData(data,NSplits,k)
%k : number of labeled samples per class
%NSplits : number of random splits

labeledunlabeledData = cell(NSplits,2);

%arrange of train data
for i=1:NSplits
    list=[];
    C=max(data{i,1}.gnd);% C= number of Classes



  list=zeros(round(2*(length(data{i,1}.gnd)/C)),C);
    for j= 1:C
        temp= find(data{i,1}.gnd ==j);
       for s= 1:size(temp,1)
           list(s,j) =temp(s);
       end
    end


    
    labeledunlabeledData{i,1}.labeledfea=[];
    %labeledunlabeledData{i,1}.labeledfea=zeros(size(data{i,1}.fea,1),k);
    %Assign labeled_train set
    
     for j=1:C
        for s=1:k
            labeledunlabeledData{i,1}.labeledfea=[labeledunlabeledData{i,1}.labeledfea , data{i,1}.fea(:,list(s,j))];
        
            end  
        labeledunlabeledData{i,1}.labeledgnd(1:k,j)=j;
        
        end
    
    labeledunlabeledData{i,1}.labeledgnd=labeledunlabeledData{i,1}.labeledgnd(:);

    
    %Assign unlabeled_train set &
    
    labeled_idx=[];
   
        temp=list(1:k,1:C);
        temp=temp(:);
        labeled_idx=sort(temp(:,1),'ascend');
    

    
    labeledunlabeledData{i,1}.unlabeledfea= data{i,1}.fea;
    labeledunlabeledData{i,1}.unlabeledfea(:,labeled_idx)=[];
    labeledunlabeledData{i,1}.unlabeledgnd= data{i,1}.gnd;
    labeledunlabeledData{i,1}.unlabeledgnd(labeled_idx)=[];
    
    
    %arrange Labeled data then unlabeled data in one matrix
    labeledunlabeledData{i,1}.arrangedfea= [labeledunlabeledData{i,1}.labeledfea,labeledunlabeledData{i,1}.unlabeledfea];
    labeledunlabeledData{i,1}.arrangedgnd= [labeledunlabeledData{i,1}.labeledgnd;labeledunlabeledData{i,1}.unlabeledgnd];
    
    
end

%keep  Test data

for i=1:NSplits
    
    labeledunlabeledData{i,2}.fea=data{i,2}.fea;
    %Keep labeled_test set
    
    labeledunlabeledData{i,2}.gnd=data{i,2}.gnd;
    
end


end


