import tensorflow as tf
import tensornets as nets
import cv2
import numpy as np
import time

from ..baseModel import BaseModel
import message_pb2

class Model(BaseModel):
    def __init__(self):
        super(Model, self).__init__()
        self.name = 'yolov3'

        self.confidence = 0.4
        self.close_sess = False

        # Define options
        self.options = ('confidence',)
        self.buttons = ('close_sess',)

        # Define inputs/outputs
        self.inputs = {'input': 3}
        self.outputs = {'output': 3}

        # start tf session with model on init so each frame is as fast as possible
        self.input_placeholder = tf.placeholder(tf.float32, [None, 416, 416, 3])
        self.model = nets.YOLOv3COCO(self.input_placeholder, nets.Darknet19)
        self.sess = tf.Session()
        self.sess.run(self.model.pretrained())
        
    def inference(self, frame_list):
        """Do an inference on the model with a set of inputs.

        # Arguments:
            frame_list: The input frame list

        Return the result of the inference.
        """

        if self.close_sess:
            self.sess.close()
            return None

        frame = frame_list[0]
        frame = self.linear_to_srgb(frame)
        frame = (frame * 255).astype(np.uint8)
        
        img=cv2.resize(frame,(416,416))
        imge=np.array(img).reshape(-1,416,416,3)

        start_time=time.time()
        preds = self.sess.run(self.model.preds, {self.input_placeholder: self.model.preprocess(imge)})
        print("--- %s seconds ---" % (time.time() - start_time)) 

        boxes = self.model.get_boxes(preds, imge.shape[1:3])
        boxes1=np.array(boxes)
        
        coco_names = ['person','bicycle','car','motorbike','aeroplane','bus','train','truck','boat',
                      'traffic light','fire hydrant','stop sign','parking meter','bench','bird',
                      'cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe',
                      'backpack','umbrella','handbag','tie','suitcase','frisbee','skis','snowboard',
                      'sports ball','kite','baseball bat','baseball glove','skateboard','surfboard',
                      'tennis racket','bottle','wine glass','cup','fork','knife','spoon','bowl',
                      'banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza',
                      'donut','cake','chair','sofa','pottedplant','bed','diningtable','toilet',
                      'tvmonitor','laptop','mouse','remote','keyboard','cell phone','microwave',
                      'oven','toaster','sink','refrigerator','book','clock','vase','scissors',
                      'teddy bear','hair drier','toothbrush']
                      
        for n in range(len(boxes1)):
            for i in range(len(boxes1[n])):
                box = boxes1[n][i]
                if boxes1[n][i][4] >= self.confidence:
                    cv2.rectangle(img,(box[0],box[1]),(box[2],box[3]),(0,255,0),1)
                    label = coco_names[n] + str(i) + str(boxes1[n][i][4])
                    cv2.putText(img, label, (box[0],box[1]), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), lineType=cv2.LINE_AA)
                    
        processed = cv2.resize(img,frame_list[0].shape[:2])
        processed = processed.astype(np.float32) / 255.
        processed = self.srgb_to_linear(processed)

        return [processed]
