import cv2
import numpy as np

image = cv2.imread("./images/test.png")

# brief
brief = cv2.xfeatures2d.BriefDescriptorExtractor().create()

# surf
sift = cv2.SIFT.create()
sift_kps = sift.detect(image)
sift_des = sift.compute(image, sift_kps)
sift_image = cv2.drawKeypoints(image, sift_kps, outImage=None)

brief_des = brief.compute(image, sift_kps)

# !!! surf, patented.
# surf = cv2.xfeatures2d.SURF.create()
# surf_kps = surf.detect(image)
# surf_image = cv2.drawKeypoints(image, surf_kps, outImage=None)

# kaze
kaze = cv2.KAZE.create()
kaze_kps = kaze.detect(image)
kaze_image = cv2.drawKeypoints(image, kaze_kps, outImage=None)

# orb
orb = cv2.ORB.create()
orb_kps = orb.detect(image)
orb_desc = orb.compute(image, orb_kps)
orb_image = cv2.drawKeypoints(image, orb_kps, outImage=None)

# kaze
kaze = cv2.KAZE.create()
kaze_kps = kaze.detect(image)
kaze_des = kaze.compute(image, kaze_kps)

# show
images = np.hstack([image, sift_image, kaze_image,orb_image])

cv2.imshow("test", images),cv2.waitKey(0),cv2.destroyAllWindows()