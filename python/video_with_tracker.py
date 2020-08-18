import os
import sys

import cv2
import dlib

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

    trackers = []
    frameCount = 0
    while cap.isOpened():
        ret, frame = cap.read()

        if frameCount == 0 or not len(trackers) or frameCount % 50 == 0:
            trackers = []
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            rects = detector.detectMultiScale(gray, scaleFactor = scaleFactor,
                    minNeighbors = minNeighbors, minSize = minSize)

            for i, (x, y, w, h) in enumerate(rects):
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                '''
                cv2.putText(frame, 'Cat #{}'.format(i + 1), (x, y - 10)
                            , cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
                '''
                tracker = dlib.correlation_tracker()
                rect = dlib.rectangle(x, y, x + w, y + h)
                tracker.start_track(frame, rect)
                trackers.append(tracker)
        else:
            for tracker in trackers:
                tracker.update(frame)
                pos = tracker.get_position()
                startX = int(pos.left())
                startY = int(pos.top())
                endX = int(pos.right())
                endY = int(pos.bottom())
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)

        cv2.imshow('Cat Face Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        frameCount += 1

    cap.release()
    cv2.destroyAllWindows()

    print('Done')
