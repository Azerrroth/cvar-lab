# -*- coding: utf-8 -*-

import numpy as np
import cv2
import time
import os

path = "./resource"
img1 = "left2.jpeg"
img2 = "right2.jpeg"


class Stitcher:
    def __init__(self):
        pass

    def stitch(self, images, ratio=0.75, reprojThresh=4.0, showMatches=False):
        '''
        Detect keypoints and extract local invariant descriptors form the given images.

        :param images: list of images to stitch
        :param ratio: ratio of the keypoint descriptor distance
        :param reprojThresh: reprojection threshold
        :param showMatches: show the matches

        :return: the stitched image
        '''
        (imageB, imageA) = images
        start = time.time()

        (kpsA, featuresA) = self.detectAndDescribe(imageA)
        end = time.time()
        print('%.5f s' % (end - start))

        (kpsB, featuresB) = self.detectAndDescribe(imageB)

        start = time.time()
        M = self.matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio,
                                reprojThresh)
        end = time.time()
        print('%.5f s' % (end - start))

        # if the match is None, then there aren't enough matched
        # keypoints to create a panorama
        if M is None:
            return None

        # otherwise, apply a perspective warp to stitch the images
        # together
        (matches, H, status) = M
        start = time.time()
        result = cv2.warpPerspective(
            imageA, H, (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
        result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB
        end = time.time()
        print('%.5f s' % (end - start))

        # check to see if the keypoint matches should be visualized
        if showMatches:
            start = time.time()
            vis = self.drawMatches(imageA, imageB, kpsA, kpsB, matches, status)
            end = time.time()
            print('%.5f s' % (end - start))
            # return a tuple of the stitched image and the
            # visualization
            return (result, vis)

        # return the stitched image
        return result

    #接收照片，检测关键点和提取局部不变特征
    #用到了高斯差分（Difference of Gaussian (DoG)）关键点检测，和SIFT特征提取
    #detectAndCompute方法用来处理提取关键点和特征
    #返回一系列的关键点
    def detectAndDescribe(self, image):
        '''
        Detect keypoints and extract local invariant descriptors form the given image.

        :param image: the image to extract features from

        :return: the keypoints and descriptors
        '''
        # convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # check to see if we are using OpenCV 3.X
        # if self.isv3:
        # detect and extract features from the image
        descriptor = cv2.SIFT_create()
        (kps, features) = descriptor.detectAndCompute(image, None)

        # convert the keypoints from KeyPoint objects to NumPy
        # arrays
        kps = np.float32([kp.pt for kp in kps])

        # return a tuple of keypoints and features
        return (kps, features)

    def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB, ratio,
                       reprojThresh):
        '''
        Match the keypoints between the two images.

        :param kpsA: keypoints from the first image
        :param kpsB: keypoints from the second image
        :param featuresA: features from the first image
        :param featuresB: features from the second image
        :param ratio: ratio of the keypoint descriptor distance
        :param reprojThresh: reprojection threshold

        :return: the matches, homography matrix, and status of the matches
        '''
        # compute the raw matches and initialize the list of actual
        # matches
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
        matches = []

        # loop over the raw matches
        for m in rawMatches:
            # ensure the distance is within a certain ratio of each
            # other (i.e. Lowe's ratio test)
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))

        # computing a homography requires at least 4 matches
        if len(matches) > 4:
            # construct the two sets of points
            ptsA = np.float32([kpsA[i] for (_, i) in matches])
            ptsB = np.float32([kpsB[i] for (i, _) in matches])

            # compute the homography between the two sets of points
            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
                                             reprojThresh)

            # return the matches along with the homograpy matrix
            # and status of each matched point
            return (matches, H, status)

        return None

    def drawMatches(self, imageA, imageB, kpsA, kpsB, matches, status):
        # initialize the output visualization image
        (hA, wA) = imageA.shape[:2]
        (hB, wB) = imageB.shape[:2]
        vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
        vis[0:hA, 0:wA] = imageA
        vis[0:hB, wA:] = imageB

        # loop over the matches
        for ((trainIdx, queryIdx), s) in zip(matches, status):
            # only process the match if the keypoint was successfully
            # matched
            if s == 1:
                # draw the match
                ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
                ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
                cv2.line(vis, ptA, ptB, (0, 255, 0), 1)

        # return the visualization
        return vis


if __name__ == '__main__':
    imageA = cv2.imread(os.path.join(path, img1))
    imageB = cv2.imread(os.path.join(path, img2))

    start = time.time()
    stitcher = Stitcher()
    (result, vis) = stitcher.stitch([imageA, imageB], showMatches=True)
    # show the images
    end = time.time()
    print('%.5f s' % (end - start))

    cv2.imwrite('./vis2.jpg', vis)
    cv2.imwrite('./result2.jpg', result)