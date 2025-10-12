# import the necessary packages
import numpy as np
import cv2
import os
import glob

class FeatureBlockBinaryPixelSum:
    def __init__(self, targetSize=(30, 15), blockSizes=((5, 5),)):
        # store the target size of the image to be described along with the set of block sizes
        self.targetSize = targetSize
        self.blockSizes = blockSizes
    
    def extract_pixel_features(self, image):
        raise Exception("This functionality is not implemented.")
        
    def extract_image_features(self, image):
        # resize the image to the target size and initialize the feature vector
        image = cv2.resize(image, (self.targetSize[1], self.targetSize[0]))

        features = []

        # loop over the block sizes
        for (blockW, blockH) in self.blockSizes:
            """
            # loop over the image for the current block size
            for y in range(0, image.shape[0], blockH):
                for x in range(0, image.shape[1], blockW):
                    # extract the current ROI, count the total number of non-zero pixels in the
                    # ROI, normalizing by the total size of the block
                    roi = image[y:y + blockH, x:x + blockW]
                    total = cv2.countNonZero(roi) / float(roi.shape[0] * roi.shape[1])

                    # update the feature vector
                    features.append(total)
            """

            # Create a kernel of ones the size of the block
            kernel = np.ones((blockH, blockW), dtype=np.float32)

            # Apply convolution (sum of pixels in each block)
            conv_result = cv2.filter2D(image.astype(np.float32), -1, kernel, borderType=cv2.BORDER_CONSTANT)

            # Take every blockH-th row and blockW-th column to simulate stride
            block_sums = conv_result[blockH - 1::blockH, blockW - 1::blockW]

            # Normalize by block area (since cv2.countNonZero does this implicitly)
            block_sums = block_sums / float(blockH * blockW * 255)  # divide by 255 to normalize binary pixels

            # Flatten and append to features
            features.extend(block_sums.flatten())

        # return the features
        return np.array(features)
    
if __name__ == "__main__":

    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Build a path to lateral and frontal images
    example_fonts = os.path.join(script_dir, "../example_fonts")

    # Obtain all images in example_fonts
    ImageFiles=sorted(glob.glob(os.path.join(example_fonts,'*.jpg')))

    # Define our feature extractor
    FeatureExtractor = FeatureBlockBinaryPixelSum(targetSize=(30,15), blockSizes=((5, 5),))

    feature_list = []
    # Loop through the images and obtain an array of feature extractors
    for img_path in ImageFiles:

        # Read image as grayscale
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        # Convert to binary
        _, binary_img = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        # Extract image features
        features = FeatureExtractor.extract_image_features(binary_img)
        feature_list.append(features)
    
        print(f"Length of feature descriptor {os.path.basename(img_path)}: {len(features)}")

    print("Feature Descriptor Obtained from the first image:", feature_list[0])    