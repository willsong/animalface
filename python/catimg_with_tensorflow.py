import glob
import os
import sys

import cv2
import numpy as np
import tensorflow.compat.v1 as tf

if __name__ == '__main__':

    if len(sys.argv) != 3:
        print('Usage: {0} /path/to/tensorflow/model.pb /path/to/test/dir'.format(sys.argv[0]))
        sys.exit(0)

    modelFile = sys.argv[1]
    testDir = sys.argv[2]

    if not os.path.isfile(modelFile):
        print('Model file not found: {0}'.format(modelFile))
        sys.exit(0)

    if not os.path.isdir(testDir):
        print('Test dir not found: {0}'.format(testDir))
        sys.exit(0)

    with tf.gfile.FastGFile(modelFile, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    print('Model loaded')

    files = glob.glob(os.path.join(testDir, "*.jp*"))
    numHits = 0
    for f in files:
        img = cv2.imread(f)

        with tf.Session() as sess:
            # Restore session
            sess.graph.as_default()
            tf.import_graph_def(graph_def, name='')

            rows = img.shape[0]
            cols = img.shape[1]
            inp = cv2.resize(img, (300, 300))
            inp = inp[:, :, [2, 1, 0]]  # BGR2RGB

            # Run the model
            out = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
                            sess.graph.get_tensor_by_name('detection_scores:0'),
                            sess.graph.get_tensor_by_name('detection_boxes:0'),
                            sess.graph.get_tensor_by_name('detection_classes:0')],
                   feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)})

            num_detections = int(out[0][0])
            for i in range(num_detections):
                classId = int(out[3][0][i])
                score = float(out[1][0][i])
                bbox = [float(v) for v in out[2][0][i]]
                if score > 0.3:
                    x = bbox[1] * cols
                    y = bbox[0] * rows
                    right = bbox[3] * cols
                    bottom = bbox[2] * rows
                    cv2.rectangle(img, (int(x), int(y)), (int(right), int(bottom)), (125, 255, 51), thickness=2)
                    cv2.putText(img, 'Cat #{}'.format(i + 1), (int(x), int(y) - 10)
                                , cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

            cv2.imshow('Cat Faces', img)
            found = True
            cv2.waitKey(0)

        if found:
            numHits += 1

    print('Num files: {0} num hits: {1}'.format(len(files), numHits))
    print('Done')
