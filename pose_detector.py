import tensorflow as tf
import numpy as np
import cv2

from mtcnn.mtcnn import MTCNN

font = cv2.FONT_HERSHEY_COMPLEX # Text in video
font_size = 0.4
blue = (225,0,0)
green = (0,128,0)
red = (0,0,255)
orange = (0,140,255)

class PoseDetector:
    def __init__(self, retinafaceModelPath) -> None:
        self._mtcnn_detector = MTCNN()
        self._retinaface_model = tf.saved_model.load(retinafaceModelPath)
        self._bbs = []

    def annotated_frame(self):
        return self._output_frame

    def bounding_boxes(self):
        return self._bbs

    def refresh(self, frame):
        self._frame = frame.copy()
        self._analyze()
    
    def _analyze(self):
        frame = self._frame
        frame = cv2.flip(frame, 1)
       
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pointss_all, bbs_all, scores_all, _ = self._face_detector(frame_rgb, image_shape_max=640, score_min=0.95, pixel_min=20, pixel_max=1000, Ain_min=90)
    
        self._values_dict = {}
        if len(pointss_all) == 0 or len(bbs_all) == 0 or len(scores_all) == 0:
            # face detection
            bbs_all, pointss_all = self._mtcnn_detector.detect_faces(frame_rgb)
            if len(pointss_all) == 0 or len(bbs_all) == 0:
                self._output_frame = frame
                return
        else:
            bbs_all = np.insert(bbs_all,bbs_all.shape[1],scores_all,axis=1)
            pointss_all = np.transpose(pointss_all)
    
        bbs = bbs_all.copy()
        pointss = pointss_all.copy()
    
        # if at least one face is detected
        if len(bbs_all) > 0:
            # process only one face (center ?)  
            bb, points = self._pick_one_face(frame, bbs, pointss)
        
            # draw land marks on face
            self._draw_landmarks(frame, bb, points)

            Xfrontal, Yfrontal = self._find_pose(points)
            self._values_dict = {
                "roll": self._find_roll(points),
                "yaw": self._find_yaw(points),
                "pitch": self._find_pitch(points),
                "x_frontal": Xfrontal,
                "y_frontal": Yfrontal
              }

            cv2.putText(frame, "Roll: {0:.2f} (-50 to +50)".format(self._values_dict["roll"]), (10,90), font, font_size, red, 1)  
            cv2.putText(frame, "Yaw: {0:.2f} (-100 to +100)".format(self._values_dict["yaw"]), (10,100), font, font_size, red, 1)
            cv2.putText(frame, "Pitch: {0:.2f} (0 to 4)".format(self._values_dict["pitch"]), (10,110), font, font_size, red, 1)
            # cv2.putText(frame, "smiles: {}, neutrals: {}, idframes: {}".format(Nsmiles, Nneutrals, Nframesperid), (10,460), font, font_size, blue, 1)
            
            cv2.putText(frame, "Xfrontal: {0:.2f}".format(self._values_dict["x_frontal"]), (10,130), font, font_size, red, 1)
            cv2.putText(frame, "Yfrontal: {0:.2f}".format(self._values_dict["y_frontal"]), (10,140), font, font_size, red, 1)   

        self._output_frame = frame
        self._bbs = bbs

    def _pick_one_face(self, frame, bbs, pointss):
        # process only one face (center ?)
        offsets = [(bbs[:,0]+bbs[:,2])/2-frame.shape[1]/2,
                (bbs[:,1]+bbs[:,3])/2-frame.shape[0]/2]
        offset_dist = np.sum(np.abs(offsets),0)
        index = np.argmin(offset_dist)
        bb = bbs[index]
        points = pointss[:,index]
        return bb, points
            
    def _draw_landmarks(self, frame, bb, points):
        # draw rectangle and landmarks on face
        cv2.rectangle(frame, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), orange, 2)
        # eyes
        # cv2.circle(img, center, radius, color, thickness=1, lineType=8, shift=0) → None
        cv2.circle(frame, (int(points[0]), int(points[5])), 2, (255,0,0), 2)
        cv2.circle(frame, (int(points[1]), int(points[6])), 2, (255,0,0), 2)
        # nose
        cv2.circle(frame, (int(points[2]), int(points[7])), 2, (255,0,0), 2)
        # mouth
        cv2.circle(frame, (int(points[3]), int(points[8])), 2, (255,0,0), 2)
        cv2.circle(frame, (int(points[4]), int(points[9])), 2, (255,0,0), 2)
        
    def _find_roll(self, pts):
        return pts[6] - pts[5]

    def _find_yaw(self, pts):
        le2n = pts[2] - pts[0]
        re2n = pts[1] - pts[2]
        return le2n - re2n

    def _find_pitch(self, pts):
        eye_y = (pts[5] + pts[6]) / 2
        mou_y = (pts[8] + pts[9]) / 2
        e2n = eye_y - pts[7]
        n2m = pts[7] - mou_y
        return e2n/n2m

    def _find_pose(self, points):
        X = points[0:5]
        Y = points[5:10]

        angle = np.arctan((Y[1]-Y[0])/(X[1]-X[0]))/np.pi*180
        alpha = np.cos(np.deg2rad(angle))
        beta = np.sin(np.deg2rad(angle))
        
        # rotated points
        Xr = np.zeros((5))
        Yr = np.zeros((5))
        for i in range(5):
            Xr[i] = alpha*X[i]+beta*Y[i]+(1-alpha)*X[2]-beta*Y[2]
            Yr[i] = -beta*X[i]+alpha*Y[i]+beta*X[2]+(1-alpha)*Y[2]

        # average distance between eyes and mouth
        dXtot = (Xr[1]-Xr[0]+Xr[4]-Xr[3])/2
        dYtot = (Yr[3]-Yr[0]+Yr[4]-Yr[1])/2

        # average distance between nose and eyes
        dXnose = (Xr[1]-Xr[2]+Xr[4]-Xr[2])/2
        dYnose = (Yr[3]-Yr[2]+Yr[4]-Yr[2])/2

        # relative rotation 0% is frontal 100% is profile
        Xfrontal = np.abs(np.clip(-90+90/0.5*dXnose/dXtot,-90,90))
        Yfrontal = np.abs(np.clip(-90+90/0.5*dYnose/dYtot,-90,90))

        # horizontal and vertical angles
        return Xfrontal, Yfrontal

    def values_dict(self):
      return self._values_dict

    def _face_detector(self, image, image_shape_max=640, score_min=None, pixel_min=None, pixel_max=None, Ain_min=None):
        '''
        Performs face detection using retinaface method with speed boost and initial quality checks based on whole image size
        
        Parameters
        ----------
        image : uint8
            image for face detection.
        image_shape_max : int, optional
            maximum size (in pixels) of image. The default is None.
        score_min : float, optional
            minimum detection score (0 to 1). The default is None.
        pixel_min : int, optional
            mininmum face size based on heigth of bounding box. The default is None.
        pixel_max : int, optional
            maximum face size based on heigth of bounding box. The default is None.
        Ain_min : float, optional
            minimum area of face in bounding box. The default is None.
        Returns
        -------
        float array
            landmarks.
        float array
            bounding boxes.
        flaot array
            detection scores.
        float array
            face area in bounding box.
        '''

        image_shape = image.shape[:2]
        
        # perform image resize for faster detection    
        if image_shape_max:
            scale_factor = max([1, max(image_shape)/image_shape_max])
        else:
            scale_factor = 1
            
        if scale_factor > 1:        
            scaled_image = cv2.resize(image, (0, 0), fx=1/scale_factor, fy=1/scale_factor)
            bbs_all, points_all = self._retinaface(scaled_image)
            bbs_all[:,:4]*=scale_factor
            points_all*=scale_factor
        else:
            bbs_all, points_all = self._retinaface(image)
        
        bbs=bbs_all.copy()
        points=points_all.copy()
        
        # check detection score
        if score_min:
            mask=np.array(bbs[:,4]>score_min)
            bbs=bbs[mask]
            points=points[mask]
            if len(bbs)==0:
                return [],[],[],[]           

        # check pixel height
        if pixel_min: 
            pixel=bbs[:,3]-bbs[:,1]
            mask=np.array(pixel>pixel_min)
            bbs=bbs[mask]
            points=points[mask]
            if len(bbs)==0:
                return [],[],[],[]           

        if pixel_max: 
            pixel=bbs[:,3]-bbs[:,1]
            mask=np.array(pixel<pixel_max)
            bbs=bbs[mask]
            points=points[mask]
            if len(bbs)==0:
                return [],[],[],[]           

        # check face area in bounding box
        Ains = []
        for bb in bbs:
            Win=min(image_shape[1],bb[2])-max(0,bb[0])
            Hin=min(image_shape[0],bb[3])-max(0,bb[1])
            Abb=(bb[2]-bb[0])*(bb[3]-bb[1])
            Ains.append(Win*Hin/Abb*100 if Abb!=0 else 0)
        Ains = np.array(Ains)

        if Ain_min:
            mask=np.array(Ains>=Ain_min)
            bbs=bbs[mask]
            points=points[mask]
            Ains=Ains[mask]
            if len(bbs)==0:
                return [],[],[],[]           
        
        scores = bbs[:,-1]
        bbs = bbs[:, :4]
        
        return points, bbs, scores, Ains

    def _retinaface(self, image):

        height = image.shape[0]
        width = image.shape[1]
        
        image_pad, pad_params = self._pad_input_image(image)    
        image_pad = tf.convert_to_tensor(image_pad[np.newaxis, ...])
        image_pad = tf.cast(image_pad, tf.float32)  
    
        outputs = self._retinaface_model(image_pad).numpy()

        outputs = self._recover_pad_output(outputs, pad_params)
        Nfaces = len(outputs)
        
        bbs = np.zeros((Nfaces,5))
        lms = np.zeros((Nfaces,10))
        
        bbs[:,[0,2]] = outputs[:,[0,2]]*width
        bbs[:,[1,3]] = outputs[:,[1,3]]*height
        bbs[:,4] = outputs[:,-1]
        
        lms[:,0:5] = outputs[:,[4,6,8,10,12]]*width
        lms[:,5:10] = outputs[:,[5,7,9,11,13]]*height
        
        return bbs, lms

    def _pad_input_image(self, img, max_steps=32):
        """pad image to suitable shape"""
        img_h, img_w, _ = img.shape

        img_pad_h = 0
        if img_h % max_steps > 0:
            img_pad_h = max_steps - img_h % max_steps

        img_pad_w = 0
        if img_w % max_steps > 0:
            img_pad_w = max_steps - img_w % max_steps

        padd_val = np.mean(img, axis=(0, 1)).astype(np.uint8)
        img = cv2.copyMakeBorder(img, 0, img_pad_h, 0, img_pad_w,
                                cv2.BORDER_CONSTANT, value=padd_val.tolist())
        pad_params = (img_h, img_w, img_pad_h, img_pad_w)

        return img, pad_params

    def _recover_pad_output(self, outputs, pad_params):
        """recover the padded output effect"""
        img_h, img_w, img_pad_h, img_pad_w = pad_params
        recover_xy = np.reshape(outputs[:, :14], [-1, 7, 2]) * \
            [(img_pad_w + img_w) / img_w, (img_pad_h + img_h) / img_h]
        outputs[:, :14] = np.reshape(recover_xy, [-1, 14])

        return outputs
