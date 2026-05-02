from fer import FER
import cv2

detector = FER(mtcnn=True)

img = cv2.imread("test.jpg")  # use any clear face image
result = detector.detect_emotions("data/test/angry/PrivateTest_88305.jpg")

print(result)