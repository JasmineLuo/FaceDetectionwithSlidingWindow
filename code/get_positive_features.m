% Starter code prepared by James Hays for CS 143, Brown University
% This function should return all positive training examples (faces) from
% 36x36 images in 'train_path_pos'. Each face should be converted into a
% HoG template according to 'feature_params'. For improved performance, try
% mirroring or warping the positive training examples.

function features_pos = get_positive_features(train_path_pos, feature_params)
% 'train_path_pos' is a string. This directory contains 36x36 images of
%   faces
% 'feature_params' is a struct, with fields
%   feature_params.template_size (probably 36), the number of pixels
%      spanned by each train / test template and
%   feature_params.hog_cell_size (default 6), the number of pixels in each
%      HoG cell. template size should be evenly divisible by hog_cell_size.
%      Smaller HoG cell sizes tend to work better, but they make things
%      slower because the feature dimensionality increases and more
%      importantly the step size of the classifier decreases at test time.


% 'features_pos' is N by D matrix where N is the number of faces and D
% is the template dimensionality, which would be
%   (feature_params.template_size / feature_params.hog_cell_size)^2 * 31
% if you're using the default vl_hog parameters

% Useful functions:
% vl_hog, HOG = VL_HOG(IM, CELLSIZE)
%  http://www.vlfeat.org/matlab/vl_hog.html  (API)
%  http://www.vlfeat.org/overview/hog.html   (Tutorial)
% rgb2gray

image_files = dir( fullfile( train_path_pos, '*.jpg') ); %Caltech Faces stored as .jpg
num_images = length(image_files);
features_pos=zeros(2*num_images,(feature_params.template_size / feature_params.hog_cell_size)^2 * 31);
% features_pos=zeros(num_images,(feature_params.template_size / feature_params.hog_cell_size)^2 * 31);
perm = vl_hog('permutation') ;
for i=1:1:num_images
    IM=single(imread(image_files(i).name));%% to be same with run_detector
    HOG1 = vl_hog(IM, feature_params.hog_cell_size);
    features_pos(2*i-1,:)=reshape(HOG1,1,[]);
%     features_pos(i,:)=reshape(HOG1,1,[]);
    HOG2 = HOG1(:,end:-1:1,perm);
    %apply mirrored hog to positive samples
%     features_pos(2*i,:)=reshape(HOG2,1,[]);
%     Lx=size(HOG1,2);
%     Ly=size(HOG1,1);
%     HOG2=HOG1;
%     for k=1:1:Ly
%         for n=1:1:floor(Lx/2)
%             temp=HOG2(k,n);
%             HOG2(k,n,:)=HOG2(k,Lx+1-n,perm);
%             HOG2(k,Lx+1-n)=temp;
%         end
%     end
    features_pos(2*i,:)=reshape(HOG2,1,[]);
end

save('proj5trafea1.mat','features_pos');
% placeholder to be deleted
%features_pos = rand(100, (feature_params.template_size / feature_params.hog_cell_size)^2 * 31);