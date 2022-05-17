

% run the below command to call the functions in this model 

[NL_Rate,L_Rate,Test_Rate]=SemiSup_AWJSLDE_Rates([1e10],[1e-9],[1e9],16,'ext_yale',32,10,[3],'PCA',95);
[NL_Rate,L_Rate,Test_Rate]=SemiSup_AWJSLDE_Rates([1e9],[1e3],[1e9],8,'umist',15,10,[3],'PCA',95);