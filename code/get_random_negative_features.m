% Starter code prepared by James Hays for CS 143, Brown University
% This function should return negative training examples (non-faces) from
% any images in 'non_face_scn_path'. Images should be converted to
% grayscale, because the positive training data is only available in
% grayscale. For best performance, you should sample random negative
% examples at multiple scales.

function features_neg = get_random_negative_features(non_face_scn_path, feature_params, num_samples)
% 'non_face_scn_path' is a string. This directory contains many images
%   which have no faces in them.
% 'feature_params' is a struct, with fields
%   feature_params.template_size (probably 36), the number of pixels
%      spanned by each train / test template and
%   feature_params.hog_cell_size (default 6), the number of pixels in each
%      HoG cell. template size should be evenly divisible by hog_cell_size.
%      Smaller HoG cell sizes tend to work better, but they make things
%      slower because the feature dimensionality increases and more
%      importantly the step size of the classifier decreases at test time.
% 'num_samples' is the number of random negatives to be mined, it's not
%   important for the function to find exactly 'num_samples' non-face
%   features, e.g. you might try to sample some number from each image, but
%   some images might be too small to find enough.

% 'features_neg' is N by D matrix where N is the number of non-faces and D
% is the template dimensionality, which would be
%   (feature_params.template_size / feature_params.hog_cell_size)^2 * 31
% if you're using the default vl_hog parameters

% Useful functions:
% vl_hog, HOG = VL_HOG(IM, CELLSIZE)
%  http://www.vlfeat.org/matlab/vl_hog.html  (API)
%  http://www.vlfeat.org/overview/hog.html   (Tutorial)
% rgb2gray

image_files = dir( fullfile( non_face_scn_path, '*.jpg' ));
num_images = length(image_files);
neg_per_img = ceil(num_samples/num_images);
% use this average to get samples uniformly
% since the imcomplete cover of image can drop the accuracy
dimension = (feature_params.template_size / feature_params.hog_cell_size)^2 * 31;
%length of descript
tempsize = feature_params.template_size;
cellsize = feature_params.hog_cell_size;
features_neg = zeros(num_images, dimension);

for n = 1:num_images
    img = im2single(rgb2gray(imread(strcat(non_face_scn_path, '/', image_files(n).name))));
    [yran, xran] = size(img);
    for s = 1:neg_per_img
        % loop for each feature in current image
        cx = ceil((xran-tempsize)*rand());
        cy = ceil((yran-tempsize)*rand());
        % rand can be replaced by multiscale features, but here I didn't
        % find good multiscale results
        scope = img(cy:cy+tempsize-1, cx:cx+tempsize-1);
        % a random fraction of current image
        features_neg(((n-1)*neg_per_img + s),:) = reshape(vl_hog(scope, cellsize), 1, dimension);
    end
end
%%here I tried several multiscale samples but didn't improve the result