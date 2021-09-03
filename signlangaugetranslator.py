from slt_ui import Ui_MainWindow
from sptotext import stt
import sys

'''PyQt5 Imports'''
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5 import QtCore
from PyQt5 import QtWidgets
from PyQt5.QtCore import *
from PyQt5.uic import loadUi
'''End of Pyqt5 imports'''

'''Detection Imports'''
import cv2
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format
import os
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
import numpy as np
'''End of detection related imports'''

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.Worker1 = Worker1()
        self.outputed_text = ''
        self.recieved = []
        self.alphabets = ''

        self.Worker1.ImageUpdate.connect(self.ImageUpdateSlot)
        self.ui.startTranslation.clicked.connect(self.Worker1.start)
        self.ui.startTranslation.clicked.connect(self.clear_field)
        self.ui.stopTranslation.clicked.connect(self.CancelFeed)
        self.ui.startSpeechToText.clicked.connect(self.speechRecognition)
        self.ui.exitButton.clicked.connect(self.exitApplication)
        self.ui.MainWidget.setGraphicsEffect(QtWidgets.QGraphicsDropShadowEffect(blurRadius = 25, xOffset =0, yOffset =0))
        
        '''Custom css for Buttons'''
        self.ui.startTranslation.setStyleSheet("QPushButton"
                             "{"
                             "background-color: rgb(217, 64, 117);"
                             "color: white;"
                             "border-radius: 10px;"
                             "border: 2px groove gray;"
                             "border-style: outset;"
                             "}"
                             "QPushButton::pressed"
                             "{"
                             "background-color : grey;"
                             "}"
                             )
        self.ui.stopTranslation.setStyleSheet("QPushButton"
                             "{"
                             "background-color: rgb(47, 79, 144);"
                             "color: white;"
                             "border-radius: 10px;"
                             "border: 2px groove gray;"
                             "border-style: outset;"
                             "}"
                             "QPushButton::pressed"
                             "{"
                             "background-color : grey;"
                             "}"
                             )
        self.ui.startSpeechToText.setStyleSheet("QPushButton"
                             "{"
                             "background-color: rgb(217, 64, 117);"
                             "color: white;"
                             "border-radius: 10px;"
                             "border: 2px groove gray;"
                             "border-style: outset;"
                             "}"
                             "QPushButton::pressed"
                             "{"
                             "background-color : grey;"
                             "}"
                             )
        self.ui.exitButton.setStyleSheet("QPushButton"
                             "{"
                             "background-color:rgb(128, 163, 0);"
                             "color: white;" 
                             "border-top-left-radius:5px;"
                             "border-top-right-radius:50px;"
                             "border-bottom-left-radius:20px;"
                             "border-bottom-right-radius:5px;"
                             "border-color:rgb(166, 166, 166);"
                             "border: 2px groove gray;"
                             "font: 63 12pt;"
                             "border-top:none;"
                             "border-right:none;"
                             "}"
                             "QPushButton::pressed"
                             "{"
                             "background-color:rgb(227, 158, 144);"
                             "}"
                             )
        '''End'''


        """Helping Hints"""
        self.ui.startTranslation.setToolTip("Start Webcam.")
        self.ui.stopTranslation.setToolTip("Stop Webcam.")
        self.ui.startSpeechToText.setToolTip("Start Microphone for 5 seconds.")
        self.ui.deafMutePersonText.setToolTip("This is for translation.")
        self.ui.hearingAblePersonText.setToolTip("This is for speech.")
        self.ui.deafMutePersonText.setText("This is for the detected gestures..") 
        self.ui.hearingAblePersonText.setText("This is for the converted speech..")
        """End"""
        
    def mousePressEvent(self, event):
        self.oldPos = event.globalPos()

    def mouseMoveEvent(self, event):
        delta = QPoint (event.globalPos() - self.oldPos)
        self.move(self.x() + delta.x(), self.y() + delta.y())
        self.oldPos = event.globalPos()
    
    def clear_field(self):
        self.ui.deafMutePersonText.setText('')
    def ImageUpdateSlot(self, Image):
        self.ui.webCam.setPixmap(QPixmap.fromImage(Image))
        self.recieved.append(self.Worker1.getHistory())
        for word in self.recieved:
            if word in self.outputed_text:
                continue
            else:
                self.outputed_text += word 
                self.outputed_text += ' ' 
                self.ui.deafMutePersonText.setText(self.outputed_text)
    def exitApplication(self):
        sys.exit()
    def CancelFeed(self):
        self.Worker1.stop()
    def speechRecognition(self):
        self.conversion = stt()
        self.ui.hearingAblePersonText.setText(self.conversion)
class Worker1(QThread):
    ImageUpdate = pyqtSignal(QImage)
    def run(self):
        self.ThreadActive = True
        self.recievedword = ''
        WORKSPACE_PATH = 'Tensorflow/workspace'
        SCRIPTS_PATH = 'Tensorflow/scripts'
        APIMODEL_PATH = 'Tensorflow/models'
        ANNOTATION_PATH = WORKSPACE_PATH+'/annotations'
        IMAGE_PATH = WORKSPACE_PATH+'/images'
        MODEL_PATH = WORKSPACE_PATH+'/models'
        PRETRAINED_MODEL_PATH = WORKSPACE_PATH+'/pre-trained-models'
        CONFIG_PATH = MODEL_PATH+'/my_ssd_mobnet/pipeline.config'
        CHECKPOINT_PATH = MODEL_PATH+'/my_ssd_mobnet/'
# Load pipeline config and build a detection model
        configs = config_util.get_configs_from_pipeline_file(CONFIG_PATH)
        detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
        ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
        ckpt.restore(os.path.join(CHECKPOINT_PATH, 'ckpt-51')).expect_partial()
        @tf.function
        def detect_fn(image):
            image, shapes = detection_model.preprocess(image)
            prediction_dict = detection_model.predict(image, shapes)
            detections = detection_model.postprocess(prediction_dict, shapes)
            return detections

        category_index = label_map_util.create_category_index_from_labelmap(ANNOTATION_PATH+'/label_map.pbtxt')

        self.cap = cv2.VideoCapture(0)
        while self.ThreadActive:
            ret, frame = self.cap.read()
            converted_frame =cv2.cvtColor (frame, cv2.COLOR_BGR2RGB)
            image_np = np.array(converted_frame)

            input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
            detections = detect_fn(input_tensor)

            num_detections = int(detections.pop('num_detections'))
            detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
            detections['num_detections'] = num_detections

            detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

            label_id_offset = 1
            image_np_with_detections = image_np.copy()

            viz_utils.visualize_boxes_and_labels_on_image_array(
                        image_np_with_detections,
                        detections['detection_boxes'],
                        detections['detection_classes']+label_id_offset,
                        detections['detection_scores'],
                        category_index,
                        use_normalized_coordinates=True,
                        max_boxes_to_draw=1,
                        min_score_thresh=.9,
                        agnostic_mode=False,
                        )
            ConvertToQtFormat = QImage(image_np_with_detections.data, image_np_with_detections.shape[1], image_np_with_detections.shape[0], QImage.Format_RGB888)
            Pic = ConvertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
            self.ImageUpdate.emit(Pic)
            self.recievedword = category_index[detections['detection_classes'][np.argmax(detections['detection_scores'])]+1]['name']
    def getHistory(self):
        return self.recievedword
    def stop(self):

        self.ThreadActive = False


app = QApplication(sys.argv)
home = MainWindow()
home.setFixedSize(1099,900)
home.setWindowFlag(Qt.FramelessWindowHint)
home.setAttribute(QtCore.Qt.WA_TranslucentBackground)
home.show()
sys.exit(app.exec_())