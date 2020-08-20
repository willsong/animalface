import os
import sys

import cv2
import dlib
import numpy as np
import tensorflow.compat.v1 as tf

if __name__ == '__main__':

    if len(sys.argv) != 3:
        print('Usage: {0} /path/to/tensorflow/model.pb /path/to/video/file.mp4'.format(sys.argv[0]))
        sys.exit(0)

    modelFile = sys.argv[1]
    videoFile = sys.argv[2]

    if not os.path.isfile(modelFile):
        print('Model file not found: {0}'.format(videoFile))
        sys.exit(0)

    if not os.path.isfile(videoFile):
        print('Video file not found: {0}'.format(videoFile))
        sys.exit(0)

    print('Video file: {0}'.format(videoFile))

    cap = cv2.VideoCapture(videoFile)
    print('Cap loaded')

    with tf.gfile.FastGFile(modelFile, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    print('Model loaded')

    trackers = []
    frameCount = 0
    while cap.isOpened():
        ret, frame = cap.read()

        if frameCount == 0 or not len(trackers) or frameCount % 50 == 0:
            print('Re-sampling')
            trackers = []

            with tf.Session() as sess:
                # Restore session
                sess.graph.as_default()
                tf.import_graph_def(graph_def, name='')

                rows = frame.shape[0]
                cols = frame.shape[1]
                inp = cv2.resize(frame, (300, 300))
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
                        cv2.rectangle(frame, (int(x), int(y)), (int(right), int(bottom)), (125, 255, 51), thickness=2)

                        tracker = dlib.correlation_tracker()
                        rect = dlib.rectangle(int(x), int(y), int(right), int(bottom))
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
