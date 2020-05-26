import numpy as np
import tensorflow.compat.v1 as tf
import cv2
import sys
import math
import argparse
from PIL import Image
from utils.anchor_generator import generate_anchors
from utils.anchor_decode import decode_bbox
from utils.nms import single_class_non_max_suppression
from load_model.pytorch_loader import load_pytorch_model, pytorch_inference

#______________________________________ SOCIAL DISTANCING_______________________________________
class Detection:
    def __init__(self, path_to_ckpt):
        self.path_to_ckpt = path_to_ckpt

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.path_to_ckpt, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.default_graph = self.detection_graph.as_default()
        self.sess = tf.Session(graph=self.detection_graph)

        
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    def processFrame(self, image):
        
        image_np_expanded = np.expand_dims(image, axis=0)
        
        
        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded})
            
        
                

        im_height, im_width,_ = image.shape
        boxes_list = [None for i in range(boxes.shape[1])]
        for i in range(boxes.shape[1]):
            boxes_list[i] = (int(boxes[0,i,0] * im_height),
                        int(boxes[0,i,1]*im_width),
                        int(boxes[0,i,2] * im_height),
                        int(boxes[0,i,3]*im_width))

        return boxes_list, scores[0].tolist(), [int(x) for x in classes[0].tolist()], int(num[0])


class Social_Distancing:
    def __init__(self,object,img):
        self.threshold=0.7
        self.avgHeight = 165   
        self.centroids = []
        boxes, scores, classes, num = obj1.processFrame(img)
        
        self.pick = []
        self.img = img
        for i in range(len(boxes)):
            
            if classes[i] == 1 and scores[i] > self.threshold:
                
                box = boxes[i]
                self.pick.append(box)
                centroid = self.centroid(box[1], box[0], box[3], box[2])
                self.centroids.append(centroid)
                cv2.rectangle(img,(box[1],box[0]),(box[3],box[2]),(255,255,0),2)
                
        txt="People Count in Frame : " + str(len(self.pick))
        cv2.putText(img, txt  , (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,0.8, (155,3,2),3)
    

    
    def centroid(self,xA, yA, xB, yB):
        midpointX = (xA + xB)/2
        midpointY = (yA + yB)/2
        return [midpointX,midpointY]

    
    def distance(self,xA1, xA2, xB1, xB2, i, j):
        inf = sys.maxsize
        a = abs(xA1-xB2)
        b = abs(xA2-xB1)
        c = abs(self.centroids[i][0] - self.centroids[j][0])

        xDist = min(a if a>0 else inf, b if b>0 else inf, c)
        xDist = xDist**2
        yDist = abs(self.centroids[i][1] - self.centroids[j][1])**2
        sqDist = xDist + yDist
        return math.sqrt(sqDist)


    def checkDistancing(self):
        img = self.img
        for i in range(len(self.pick)-1):
            boxI = self.pick[i]
            (xA1, yA1, xB1, yB1) = (boxI[1], boxI[0], boxI[3], boxI[2])
            for j in range(i+1,len(self.pick)):
                boxJ = self.pick[j]
                (xA2, yA2, xB2, yB2) = (boxJ[1], boxJ[0], boxJ[3], boxJ[2])

                
                dist = self.distance(xA1, xA2, xB1, xB2, i, j)

                
                heightI = abs(yA1 - yB1)
                heightJ = abs(yA2 - yB2)

                if heightI==0 or heightJ==0:
                    continue

                ratioI = self.avgHeight/heightI     # in cm/pixels
                ratioJ = self.avgHeight/heightJ

                meanRatio = (ratioI + ratioJ)/2

                dist = dist * meanRatio       # in cm
                

                if dist<100:
                    cv2.rectangle(img,(xA1,yA1),(xB1,yB1),(0,255,255),2)
                    cv2.rectangle(img,(xA2,yA2),(xB2,yB2),(0,255,255),2)
                    cv2.putText(img, "Violation"  , (xA1 + 1, yA1 - 1),
                        cv2.FONT_HERSHEY_SIMPLEX,0.8, (255,0,0),2)

#______________________________________ MASK DETECTION_________________________________________________

def inference(image,
              conf_thresh=0.5,
              iou_thresh=0.5,
              target_shape=(360, 360),
              draw_result=True,
              show_result=False
              ):
    
    
    output_info = []
    height, width, _ = image.shape
    image_resized = cv2.resize(image, target_shape)
    image_np = image_resized / 255.0  
    image_exp = np.expand_dims(image_np, axis=0)

    image_transposed = image_exp.transpose((0, 3, 1, 2))

    y_bboxes_output, y_cls_output = pytorch_inference(model, image_transposed)
    
    y_bboxes = decode_bbox(anchors_exp, y_bboxes_output)[0]
    y_cls = y_cls_output[0]
    
    bbox_max_scores = np.max(y_cls, axis=1)
    bbox_max_score_classes = np.argmax(y_cls, axis=1)

    
    keep_idxs = single_class_non_max_suppression(y_bboxes,
                                                 bbox_max_scores,
                                                 conf_thresh=conf_thresh,
                                                 iou_thresh=iou_thresh,
                                                 )

    for idx in keep_idxs:
        conf = float(bbox_max_scores[idx])
        class_id = bbox_max_score_classes[idx]
        bbox = y_bboxes[idx]
        
        xmin = max(0, int(bbox[0] * width))
        ymin = max(0, int(bbox[1] * height))
        xmax = min(int(bbox[2] * width), width)
        ymax = min(int(bbox[3] * height), height)

        if draw_result:
            if class_id == 0:
                color = (0, 255, 0)
            else:
                color = (255, 0, 0)
                
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.putText(image, "%s: %.2f" % (id2class[class_id], conf), (xmin + 2, ymin - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color,2)
        output_info.append([class_id, conf, xmin, ymin, xmax, ymax])

    if show_result:
        Image.fromarray(image).show()
    return output_info



                                                                   
if __name__ == "__main__":
	
	model = load_pytorch_model('models/model360.pth');
	
	feature_map_sizes = [[45, 45], [23, 23], [12, 12], [6, 6], [4, 4]]
	anchor_sizes = [[0.04, 0.056], [0.08, 0.11], [0.16, 0.22], [0.32, 0.45], [0.64, 0.72]]
	anchor_ratios = [[1, 0.62, 0.42]] * 5

	
	anchors = generate_anchors(feature_map_sizes, anchor_sizes, anchor_ratios)

	
	anchors_exp = np.expand_dims(anchors, axis=0)

	id2class = {0: 'Mask', 1: 'NoMask'}
	
	
	model_path = 'faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb'   
	obj1 = Detection(path_to_ckpt=model_path)
    
	
	cap = cv2.VideoCapture('/home/sakshi/Desktop/Teach_a_Lesson.mp4')
	
	height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
	width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
	fps = cap.get(cv2.CAP_PROP_FPS)
	fourcc = cv2.VideoWriter_fourcc(*'XVID')
	total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
	
	while True :
		status, img_raw = cap.read()
		obj2 = Social_Distancing(obj1, img_raw)
		obj2.checkDistancing()
		
		img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
		
		inference(img_raw)
		cv2.imshow("Mask Detector", img_raw[:, :, ::-1])

        
		key = cv2.waitKey(1)
		if key == 27:
				break
        
	cap.release()
	cv2.destroyAllWindows()
    
        
                                                                   
                  
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
