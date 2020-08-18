import os
import sys

import cv2

if __name__ == '__main__':

    if len(sys.argv) != 3:
        print('Usage: {0} /path/to/cascade/file.xml /path/to/video/file.mp4'.format(sys.argv[0]))
        sys.exit(0)

    modelFile = sys.argv[1]
    videoFile = sys.argv[2]

    if not os.path.isfile(modelFile):
        print('Cascade file not found: {0}'.format(modelFile))
        sys.exit(0)

    if not os.path.isfile(videoFile):
        print('Video file not found: {0}'.format(videoFile))
        sys.exit(0)

    print('Cascade file: {0}'.format(modelFile))
    print('Video file: {0}'.format(videoFile))

    cap = cv2.VideoCapture(videoFile)
    print('Cap loaded')

    detector = cv2.CascadeClassifier(modelFile)
    print('Detector loaded')

    # adjust these numbers as necessary
    scaleFactor = 1.1
    minNeighbors = 3
    minSize = (100, 100)

    while cap.isOpened():
        ret, frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        rects = detector.detectMultiScale(gray, scaleFactor = scaleFactor,
                minNeighbors = minNeighbors, minSize = minSize)

        found = False
        for i, (x, y, w, h) in enumerate(rects):
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, 'Cat #{}'.format(i + 1), (x, y - 10)
                        , cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

        cv2.imshow('Cat Face Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    print('Done')
