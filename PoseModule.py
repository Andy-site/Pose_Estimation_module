import cv2
import mediapipe as mp
import time
import math

class PoseDetector:
    def __init__(self, mode=False, smooth=True, detectionCon=0.5, trackCon=0.5):
        """
        Initializes the PoseDetector with given parameters.

        :param mode: If True, uses static image mode.
        :param smooth: If True, smooths landmark tracking.
        :param detectionCon: Minimum confidence for pose detection.
        :param trackCon: Minimum confidence for landmark tracking.
        """
        self.mode = mode
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(
            static_image_mode=self.mode,
            smooth_landmarks=self.smooth,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )

    def findPose(self, img, draw=True):
        """
        Detects pose in the given image.

        :param img: The input image.
        :param draw: If True, draws landmarks on the image.
        :return: Image with detected pose landmarks.
        """
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        
        if self.results.pose_landmarks and draw:
            self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
                                        self.mpPose.POSE_CONNECTIONS)
        return img

    def findPosition(self, img, draw=True):
        """
        Finds positions of landmarks in the given image.

        :param img: The input image.
        :param draw: If True, draws circles on landmarks.
        :return: List of landmark positions.
        """
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, _ = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)  # Blue circle for landmarks
        return self.lmList

    def findAngle(self, img, p1, p2, p3, draw=True):
        """
        Calculates the angle between three points.

        :param img: The input image.
        :param p1: Index of the first landmark.
        :param p2: Index of the second landmark (vertex).
        :param p3: Index of the third landmark.
        :param draw: If True, draws lines and labels on the image.
        :return: The calculated angle.
        """
        # Get the landmarks
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]

        # Calculate the angle
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) -
                             math.atan2(y1 - y2, x1 - x2))
        if angle < 0:
            angle += 360

        # Draw the angle lines and landmarks
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
            cv2.circle(img, (x1, y1), 10, (255, 0, 0), cv2.FILLED)  # Blue circle for first landmark
            cv2.circle(img, (x2, y2), 10, (255, 0, 0), cv2.FILLED)  # Blue circle for second landmark
            cv2.circle(img, (x3, y3), 10, (255, 0, 0), cv2.FILLED)  # Blue circle for third landmark
            cv2.putText(img, str(int(angle)), (x2 - 50, y2 + 50),
                        cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)  # Blue text for angle
        return angle

def main():
    # Video capture from a file
    cap = cv2.VideoCapture('PoseVideos/1.mp4')
    pTime = 0
    detector = PoseDetector()
    
    while True:
        success, img = cap.read()
        if not success:
            break
        
        img = detector.findPose(img)
        lmList = detector.findPosition(img, draw=False)

        if len(lmList) != 0:
            # Example usage: highlight the right wrist (landmark index 14)
            cv2.circle(img, (lmList[14][1], lmList[14][2]), 15, (0, 0, 255), cv2.FILLED)  # Blue circle for right wrist

        # Calculate FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                    (0, 255, 0), 3)  # Green text for FPS

        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()
