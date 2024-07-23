# "/data/dataSetB_10k/36253_2011.jpg"

from mtcnn import MTCNN
import cv2

# Load the detector
detector = MTCNN()

# Read the image
img = cv2.imread("/data/dataSetB_10k/36253_2011.jpg", cv2.COLOR_RGB2BGR)
#img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Detect faces and landmarks
results = detector.detect_faces(img)
for result in results:
    bounding_box = result['box']
    keypoints = result['keypoints']

    cv2.rectangle(img, (bounding_box[0], bounding_box[1]),
                  (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]),
                  (0, 155, 255), 2)

    for key, point in keypoints.items():
        cv2.circle(img, point, 2, (0, 155, 255), 2)

# Display the output
cv2.imshow("Image", img)
cv2.waitKey(0)
