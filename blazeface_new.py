import tensorflow as tf
import cv2
import numpy as np
import math
import time
import os


class BlazeFace():

    def __init__(self,model_path):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_idx = self.interpreter.get_input_details()[0]["index"]
        self.regressions_idx = self.interpreter.get_output_details()[0]["index"]
        self.classifications_idx = self.interpreter.get_output_details()[1]["index"]

        # z-index first
        anchor_layer1_center = np.array([[(x + 0.5)/16,(y+0.5)/16] for y in range(16) for x in range(16) for i in range(2)])
        anchor_layer234_center = np.array([[(x + 0.5)/8,(y+0.5)/8] for y in range(8) for x in range(8) for i in range(6)])
        #np.set_printoptions(threshold=1000000)
        #print(anchor_layer1_center.shape)  (512,2)
        #print(anchor_layer234_center.shape)(384,2)
        print(self.input_idx)
        print(self.regressions_idx)
        print(self.classifications_idx)
        self.anchor_center = np.concatenate((anchor_layer1_center,anchor_layer234_center),axis=0)


    def blaze_decode(self,regressions,classifications):
        # sigmoid activation for classifications
        s_score = 1/(1+np.exp(-classifications))
        # filter
        select_idx = np.squeeze(np.greater(s_score,0.85),axis=2)
        print(select_idx.shape)
        select_s_score = s_score[select_idx]
        
        select_regressions = regressions[select_idx]
        select_idx = np.squeeze(select_idx)
        select_anchor = self.anchor_center[select_idx]

        # decode selected regressions
        select_regressions[:,4:] = select_regressions[:,4:]/128 + np.tile(select_anchor,6)
        xy_center = select_regressions[:,0:2]/128 + select_anchor
        wh = select_regressions[:,2:4] / 128
        select_regressions[:,0:2] = xy_center - wh / 2
        select_regressions[:,2:4] = xy_center + wh / 2

        # nms
        sort_idx = np.squeeze(np.argsort(select_s_score,axis=0))
        if sort_idx.size > 1:
            Area = (select_regressions[:,2] - select_regressions[:,0]) * (select_regressions[:,3] - select_regressions[:,1])
            final_idx = []
            final_idx.append(sort_idx[-1])
            while len(sort_idx) > 1:
                area_a = Area[sort_idx[:-1]]
                area_b = Area[sort_idx[-1]]
                box_a = select_regressions[sort_idx[:-1]]
                box_b = select_regressions[sort_idx[-1]]
                max_x = np.maximum(box_a[:,0],box_b[0])
                max_y = np.maximum(box_a[:,1],box_b[1]) 
                min_x = np.minimum(box_a[:,2],box_b[2])
                min_y = np.minimum(box_a[:,3],box_b[3])
                inter = (min_x-max_x) * (min_y-max_y)
                iou = inter / (area_a + area_b - inter)
                sort_idx = sort_idx[:-1][np.less(iou,0.60)]
                if len(sort_idx) > 0:
                    final_idx.append(sort_idx[-1])
            return select_regressions[final_idx],select_s_score[final_idx]
        else:
            return [],[]

    def preprocess_Image(self,src_image):
        img = cv2.cvtColor(src_image,cv2.COLOR_BGR2RGB)
        img_width = img.shape[1]
        img_height = img.shape[0]

        max_length = max(img_width,img_height)
        ratio = max_length / 128.0
        destiny_h = int(img_height / ratio)
        destiny_w = int(img_width / ratio)
        resized = cv2.resize(img,(destiny_w,destiny_h))
        
        start_h = (128-destiny_h) // 2
        start_w = (128-destiny_w) // 2
        end_h = (128-destiny_h) // 2 + destiny_h
        end_w = (128-destiny_w) // 2 + destiny_w

        d_image = np.zeros((128,128,3),np.uint8)
        d_image[start_h:end_h,start_w:end_w,:] = resized
        d_image = ((d_image-127.5)/127.5)
        d_image = np.expand_dims(d_image,axis=0)
        d_image = np.float32(d_image)
        return start_w,start_h,ratio,d_image

    #处理文件夹
    #dir_path  文件夹目录
    #type_name 图片类型
    def detect_face_handler(self,dir_path,type_name):
        self.draw = True
        self.index_adjust = 0
        list_data = os.listdir(dir_path)
        for i in list_data:
            index = i.find(type_name)#".jpg"
            temp_str_path = dir_path+'/'+ i
            if index >= 0:
                #print(temp_str_path)
                res = self.detect_save_face(temp_str_path,"./RealImageSave")
                #self.detect_show_face(temp_str_path)
                pass
        pass
    
    #显示图像
    def detect_show_face(self,image_path):
        process_start_time = time.time()
        img_bgr = cv2.imread(image_path)

        img_width = img_bgr.shape[1]
        img_height = img_bgr.shape[0]

        start_w,start_h,ratio,d_image=self.preprocess_Image(img_bgr)
        process_end_time = time.time() 

        inference_start_time = time.time()
        self.interpreter.set_tensor(self.input_idx,d_image)
        self.interpreter.invoke()
        regressions = self.interpreter.get_tensor(self.regressions_idx)
        classifications = self.interpreter.get_tensor(self.classifications_idx)
        inference_end_time = time.time()

        postprocess_start_time = time.time()
        detections,scores = self.blaze_decode(regressions,classifications)
        if len(detections) > 0:
            for item,score in zip(detections,scores):
                x_min = item[0] * 128 - start_w
                y_min = item[1] * 128 - start_h
                if x_min >=0 and y_min >= 0:
                    x_max = int((item[2]*128-start_w)*ratio) 
                    y_max = int((item[3]*128-start_h)*ratio)
                    x_min = int(x_min*ratio)
                    y_min = int(y_min*ratio)
                    if x_max <= img_width and y_max <= img_height:
                        cv2.rectangle(img_bgr, (x_min, y_min), (x_max, y_max), (255,0,0), 2)
                        cv2.putText(img_bgr, '{:.3f}'.format(float(score)), (x_min, y_min - 6)
                            , cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        for i in range(6):
                            x = int((item[3+i*2+1] * 128 - start_w) * ratio)
                            y = int((item[3+i*2+2] * 128 - start_h) * ratio)
                            cv2.circle(img_bgr,(x,y),3,(0,0,255),-1)
                        #width = x_max - x_min
                        #height = y_max - y_min
                        #return x_min,y_min,width,height
        else:
            print('no face was detected')
            return None
        
        # postprocess_end_time = time.time()
        # print('process cost:{:.2f} ms'.format((process_end_time - process_start_time)*1000))
        # print('inference cost:{:.2f} ms'.format((inference_end_time - inference_start_time)*1000))
        # print('post cost:{:.2f} ms'.format((postprocess_end_time - postprocess_start_time )*1000))

        cv2.imshow("", img_bgr)
        print("---enter  Esc  , Process Exit, Thanks ---")
        while True:
            k = cv2.waitKey(30)
            if k==27:
                break;
        cv2.destroyAllWindows()

        return None

    def adjust_rect(self,y_min,y_max,x_min,x_max):
        self.adjust_dist = 30#扩张距离
        tempxmi = x_min - self.adjust_dist
        tempymi = y_min - self.adjust_dist
        if tempxmi < 0:
            x_min = 0
        else:
            x_min = tempxmi
        if tempymi < 0:
            y_min = 0
        else:
            y_min = tempymi
        tempxmx = x_max + self.adjust_dist
        tempymx = y_max + self.adjust_dist
        if tempxmx > self.image_width:
            x_max = self.image_width
        else:
            x_max = tempxmx
        if tempymx > self.image_height:
            y_max = self.image_height
        else:
            y_max = tempymx
        if x_min == 0 or x_max == self.image_width:
            return None
        if y_min == 0 or y_max == self.image_height:
            return None 
        return y_min,y_max,x_min,x_max
        
    def detect_save_face(self,image_path,save_path):
        start_time = time.time()
        img_bgr = cv2.imread(image_path)
        if img_bgr is not None:
            img = cv2.cvtColor(img_bgr,cv2.COLOR_BGR2RGB)
        else:
            return 0

        img_width = img.shape[1]
        img_height = img.shape[0]

        self.image_width = img_width
        self.image_height = img_height

        max_length = max(img_width,img_height)
        ratio = max_length / 128.0
        destiny_h = int(img_height / ratio)
        destiny_w = int(img_width / ratio)
        resized = cv2.resize(img,(destiny_w,destiny_h))
        
        start_h = (128-destiny_h) // 2
        start_w = (128-destiny_w) // 2
        end_h = (128-destiny_h) // 2 + destiny_h
        end_w = (128-destiny_w) // 2 + destiny_w

        d_image = np.zeros((128,128,3),np.uint8)
        d_image[start_h:end_h,start_w:end_w,:] = resized
        d_image = ((d_image-127.5)/127.5)
        d_image = np.expand_dims(d_image,axis=0)
        d_image = np.float32(d_image)

        self.interpreter.set_tensor(self.input_idx,d_image)
        self.interpreter.invoke()
        regressions = self.interpreter.get_tensor(self.regressions_idx)
        classifications = self.interpreter.get_tensor(self.classifications_idx)
        detections,scores = self.blaze_decode(regressions,classifications)

        if len(detections) > 0:
            item = detections[0]
            x_min = item[0] * 128 - start_w
            y_min = item[1] * 128 - start_h
            if x_min >=0 and y_min >= 0:
                x_max = int((item[2]*128-start_w)*ratio) 
                y_max = int((item[3]*128-start_h)*ratio)
                x_min = int(x_min*ratio)
                y_min = int(y_min*ratio)
                if x_max <= img_width and y_max <= img_height:#右下角的坐标点在图像内
                    #detect_img = img_bgr[y_min:y_max,x_min:x_max]#h,w
                    #save_img = cv2.resize(detect_img,(112,112))
                    #cv2.imwrite(save_path,save_img)
                    self.index_adjust += 1#计数累加
                    save_n = "%05d" % self.index_adjust
                    save_path = save_path + '/'+ str(save_n) + '.jpg'#重新保存图片路径命名
                    adjust_restult = self.adjust_rect(y_min,y_max,x_min,x_max)
                    if adjust_restult:
                        detect_img = img_bgr[adjust_restult[0]:adjust_restult[1],adjust_restult[2]:adjust_restult[3]]#h,w
                        save_img = cv2.resize(detect_img,(256,256))
                        cv2.imwrite(save_path,save_img) 
                    return 1
                else:
                    return 0
            else:
                return 0
        else:
            return 0

    def detect_camera_face(self,camera_num):
        cap = cv2.VideoCapture(camera_num)
        print("---enter  Esc  , Process Exit, Thanks ---")
        process_times_timer =0
        process_times_timer_sum=0
        while True:
            success,frame = cap.read()
            img_width = frame.shape[1]
            img_height = frame.shape[0]

            process_start_time = time.time()
            start_w,start_h,ratio,d_image=self.preprocess_Image(frame)
            process_end_time = time.time()

            inference_start_time = time.time()
            self.interpreter.set_tensor(self.input_idx,d_image)
            self.interpreter.invoke()
            regressions = self.interpreter.get_tensor(self.regressions_idx)
            classifications = self.interpreter.get_tensor(self.classifications_idx)
            inference_end_time = time.time()

            postprocess_start_time = time.time()
            detections,scores = self.blaze_decode(regressions,classifications)
            if len(detections) > 0:
                for item,score in zip(detections,scores):
                    x_min = item[0] * 128 - start_w
                    y_min = item[1] * 128 - start_h
                    if x_min >=0 and y_min >= 0:
                        x_max = int((item[2]*128-start_w)*ratio) 
                        y_max = int((item[3]*128-start_h)*ratio)
                        x_min = int(x_min*ratio)
                        y_min = int(y_min*ratio)
                        if x_max <= img_width and y_max <= img_height:
                            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255,0,0), 2)
                            cv2.putText(frame, '{:.3f}'.format(float(score)), (x_min, y_min - 6)
                                , cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                            for i in range(6):
                                x = int((item[3+i*2+1] * 128 - start_w) * ratio)
                                y = int((item[3+i*2+2] * 128 - start_h) * ratio)
                                cv2.circle(frame,(x,y),3,(0,0,255),-1)
            postprocess_end_time = time.time()

            process_times_timer += 1
            process_times_timer_sum += (inference_end_time - inference_start_time)*1000
            if process_times_timer == 1000:
                print('Sum post cost:{:.2f} ms'.format(process_times_timer_sum/1000))
                process_times_timer_sum = 0
                process_times_timer = 0

            #print('process cost:{:.2f} ms'.format((process_end_time - process_start_time)*1000))
            #print('inference cost:{:.2f} ms'.format((inference_end_time - inference_start_time)*1000))
            #print('post cost:{:.2f} ms'.format((postprocess_end_time - postprocess_start_time )*1000))

            cv2.imshow("face", frame)
            k = cv2.waitKey(30)
            if k==27:
                break;
        cap.release()
        cv2.destroyAllWindows()
        return 0

if __name__ == "__main__":
    blazeface = BlazeFace('face_detection_front.tflite')
    blazeface.detect_face_handler('/home/work/DM-dataset/originalimages',".jpg")