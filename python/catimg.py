import glob
import os
import sys

import cv2

if __name__ == '__main__':

    if len(sys.argv) != 3:
        print('Usage: {0} /path/to/cascade/file.xml /path/to/test/dir'.format(sys.argv[0]))
        sys.exit(0)

    modelFile = sys.argv[1]
    testDir = sys.argv[2]

    if not os.path.isfile(modelFile):
        print('Cascade file not found: {0}'.format(modelFile))
        sys.exit(0)

    if not os.path.isdir(testDir):
        print('Test dir not found: {0}'.format(testDir))
        sys.exit(0)

    detector = cv2.CascadeClassifier(modelFile)
    print('Detector loaded')

    # adjust these numbers as necessary
    scaleFactor = 1.05
    minNeighbors = 3
    minSize = (25, 25)

    files = glob.glob(os.path.join(testDir, "*.jp*"))
    numHits = 0
    for f in files:
        img = cv2.imread(f)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        rects = detector.detectMultiScale(gray, scaleFactor = scaleFactor,
                minNeighbors = minNeighbors, minSize = minSize)

        found = False
        for i, (x, y, w, h) in enumerate(rects):
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(img, 'Cat #{}'.format(i + 1), (x, y - 10)
                        , cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

            cv2.imshow('Cat Faces', img)
            found = True
            cv2.waitKey(0)

        if found:
            numHits += 1

    print('Num files: {0} num hits: {1}'.format(len(files), numHits))
    print('Done')
