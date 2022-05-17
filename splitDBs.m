function data = splitDBs(DB_name , nTrnSmp , NSplits)

%%
%  This function produce random splits from databases
%   Inputs    DB_name : name of database in string format
%             nTrnSmp : number of train sample for each class
%             NSplits : number of random splits

%     Output
%                 data : is a cell contains the train and test information
%                 of each split.
%                     first column of each row of the cell contains train data
%                     second column of each row of the cell contains test data
%                       each cell of the output data is a structure, contains
%                         fea : vectorize feature, each column is an observation
%                         gnd : ground truth or label of coresponding column

%         data{k,1}.fea : features of kth train split
%         data{k,1}.gnd : ground truth of kth split
%
%         data{k,2}.fea : features of kth split
%         data{k,2}.gnd : ground truth of kth split

%   Sample
%             data = splitDBs('ext_yale' , 10 , 8)
%    8 random splits from Extended yale database with 10 train per subject
%
%             data = splitDBs('pie' , 5 , 9)
%    9 random splits from pie database with 5 train per subject
%
%%
if nargin <3
    error('Number of inputs should be 3')
end
if NSplits >10
    warning('Warn:bad_value_for_nSplits', ...
        'Number of splits should not be more than 10 \n Number of splits is set to 10');
    NSplits = 10;
end

switch lower(DB_name)
    
    case lower('Ext_Yale')
        load fadi_Yale_Ext
    case lower('ORL')
        load fadi_ORL
    case lower('Coil20')
        load fadi_coil20
    case lower('Coil20C')
        load fadi_coil20_complete
    case lower('Pie')
        load fadi_pie
        
    case lower('Pf01')
        load fadi_pf01
    case lower('usps')
        load fadi_usps_small
        
    case lower('FERET')
        load fadi_feret7
        
    case lower('Yale')
        load fadi_yale
    case lower('caltech101')
        load caltech101
        
    case lower('facepix_pos')
        load fadi_facepix_pose.mat
        
    case lower('facepixID')
        load fadi_facepix_ID_Small.mat
        
    case lower('facepix_dark')
        load fadi_facepix_dark.mat
        
    case lower('facepix_light')
        load fadi_facepix_light.mat
    case lower('sceneCroped')
        load sceneCroped.mat
    case lower('umist')
        load fadi_umist.mat
    case lower('honda')
        load fadi_honda_small.mat
        
    case lower('indoor')
        load indoor.mat
    case lower('mnist')
        load mnist_Res50.mat
        case lower('norb')
        load NORB.mat
        case lower('face')
        load face
    otherwise
        disp(['  '])
        disp('The name of database is not valid.')
        disp(' Try Yale , Ext_Yale , Feret , Pie , Pf01 , facepix_pos , facepix_ID , facepix_dark or facepix_light')
        disp(['  '])
        data=[];
        return
end

Smpls =   face.Splits;

if strcmp (nTrnSmp ,'all')
    
    if size(face.mat,2)== size(face.labels,1)
    face.labels = face.labels';
end

%%Prealocation
data = cell(NSplits,2);

for i=1:NSplits
    %Assign train set
data{i,1}.fea=zscore(double(face.mat));
data{i,1}.gnd=face.labels';

%Assing the rest as test set
data{i,2}.fea=[];
data{i,2}.gnd=[];
    
end
    
    
else
if nTrnSmp > size(Smpls,2)
    error('Error:WrongTstNumber' , ...
        ['Error in the number of images per class \n' ,...
        ' Number of train samples is more than the number of images in one class \n' , ...
        '  Try a value less than ' num2str(size(Smpls,2))] )
end

if nTrnSmp == size(Smpls,2)+1
    warning('Warn:Number_of_images_per_class', ...
        ' Number of train samples is equal to the number of images in one class \n The test data might be empty')
end



if size(face.mat,2)== size(face.labels,1)
    face.labels = face.labels';
end

%%Prealocation
data = cell(NSplits,2);

for i=1:NSplits
    Sel_Lst =    Smpls    (i , 1:nTrnSmp);
    [data{i,1} data{i,2}] = MakeTrainTestData(face,Sel_Lst);
end

end

function [traind testd] = MakeTrainTestData( face , Sel_Lst )

list=[];
Lbls=unique(face.labels);
for i=1:length(Lbls)
    tmp      = find( face.labels == Lbls(i) );
    list    =   [list  tmp(Sel_Lst) ];
end

%%
%Assign train set
traind.fea=zscore(double(face.mat(:, list )));
traind.gnd=face.labels(list)';

%Delete train set
face.mat(:,list)=[];
face.labels(list)=[];

%Assing the rest as test set
testd.fea=zscore(double(face.mat));
testd.gnd=(face.labels)';