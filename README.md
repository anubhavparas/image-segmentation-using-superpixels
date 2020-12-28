# Image Segmentation Using Superpixels

The goal is to build a segmentation network, which uses SLIC Superpixels as input. In essense, it will be a classifier for superpixels. 
The end product is a system which, when given an image, computes superpixels and classifies each superpixel as one of the 9 classes of MSRC v1.

In the project:
- A classifier network is build to classify the superpixels into one of the 10 classes (including background or void).
- To extract the features of the superpixels pretrained VGG16 network is used.
- The following steps were taken:
  - For each image:
    - get superpixels sp_i for image x. We adopt 100 segments in this assignment, 'segments = slic(image, n_segments=100, compactness=10)'
    - for every superpixel sp_i in the image,
      - find the smallest rectangle which can enclose sp_i
      - dilate the rectangle by 3 pixels.
      - get the same region from the segmentation image (from the file with similar name with *_GT). The class for this sp_i is mode of segmentation classes in that same region. Save the dilated region as npy (jpg is lossy for such small patches). Refer to the flow below:
      
      ![alt text](./images/steps.png?raw=true "Steps to follow")
      
  - Use pre-trained VGG16 network and replace the last few layers were replaced by fully connected layers to handle 10 classes.
- **Multi-resoltution network was used to try to improve the performance of the existing network** - paper:[Feedforward semantic segmentation with zoom-out features](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Mostajabi_Feedforward_Semantic_Segmentation_2015_CVPR_paper.pdf)
    1. Used the pre-trained VGG16 model to extract the feature maps
    2. As the model is pre-trained, the feature maps were extracted before the training of the model.
    3. Approach:
      - Split the MSRCv1 data into train and test
      - For each image in the training/test data:
        - Get the superpixel segmentation map using SLIC,
        - Get the upsampled feature maps by feeding the image in the VGG16 network (13 from the 13 conv layers in VGG16),
        - For the each superpixel in the segmentation map:
          - Get the corresponding region the feature maps,
          - Average pooling and converting into 1 x depth_of_ith_feature_map vector,
          - Concatenating the all the vectors to represent the feature vector of that the partiular superpixel,
          - Saving the feature vector along with the ground-truth label
      - Design a simple classifier (MLP network with hidden layers and one output layer of ten units).
      - Train the model with the feature vectors extracted from the VGGNet.
 
