
import numpy as np
import time
from openvino.inference_engine import IENetwork, IECore
import os
import cv2
import argparse
import sys


class Queue:
    '''
    Class for dealing with queues
    '''
    def __init__(self):
        self.queues=[]

    def add_queue(self, points):
        self.queues.append(points)

    def get_queues(self, image):
        for q in self.queues:
            x_min, y_min, x_max, y_max=q
            frame=image[y_min:y_max, x_min:x_max]
            yield frame
    
    def check_coords(self, coords):
        d={k+1:0 for k in range(len(self.queues))}
        for coord in coords:
            for i, q in enumerate(self.queues):
                if coord[0]>q[0] and coord[2]<q[2]:
                    d[i+1]+=1
        return d


class PersonDetect:
    '''
    Class for the Person Detection Model.
    '''

    def __init__(self, model_name, device, threshold=0.60):
        self.model_weights=model_name+'.bin'
        self.model_structure=model_name+'.xml'
        self.device=device
        self.threshold=threshold
        self.w_image = 0
        self.h_image = 0

        try:
            try:
                self.core=IECore()
                self.model=self.core.read_network(model=self.model_structure, weights=self.model_weights)
            except AttributeError:
                self.model=IENetwork(model=self.model_structure, weights=self.model_weights)
        except Exception:
            raise ValueError("Could not Initialise the network. Have you entered the correct model path?")

        self.input_name=next(iter(self.model.inputs))
        # Gets the input shape of the network
        self.input_shape=self.model.inputs[self.input_name].shape
        self.output_name=next(iter(self.model.outputs))
        # Gets the output shape of the network
        self.output_shape=self.model.outputs[self.output_name].shape

    def load_model(self):
        '''
        Loads the network 
        '''
        num_req=1
        
        self.net=self.core.load_network(network=self.model, device_name=self.device, num_requests=num_req)
    
        return self.net

        
    def predict(self, image):
        '''
        Image prepreprocessing, postprocessing and drawing bounding boxes
        '''
        request_id = 0
        
        preprocess_image = self.preprocess_input(image)
        input_dict = {self.input_name:preprocess_image}
        
        # Makes an asynchronous inference request, given an input image.
        self.net.start_async(request_id=request_id, inputs=input_dict)
        if self.net.requests[request_id].wait(-1) ==0:
            # Parse detection results to the current request
            out = self.net.requests[request_id].outputs[self.output_name]

            coords = self.preprocess_outputs(out)
            box, image = self.draw_outputs(coords, image)
            
        return box, image
              
    
    def draw_outputs(self, coords, image):
        '''
        Find and draw the bounding box
        '''
        print("draw_outputs")
        h_image = image.shape[0]
        w_image = image.shape[1]
        img_coords = []
        color = (11, 156, 254)
        for coord in coords:
            # Draw objects
                xmin = int(coord[3] * w_image)
                ymin = int(coord[4] * h_image)
                xmax = int(coord[5] * w_image)
                ymax = int(coord[6] * h_image)
                # Draw the bounding box
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
                img_coords.append([xmin, ymin, xmax, ymax])
        
        return img_coords, image
    

    def preprocess_outputs(self, outputs):
        '''
        # Draw objects only when probability is more than specified threshold
        '''
        thres_coord = []
        for coord in outputs[0][0]:
            if coord[2] > self.threshold:
                thres_coord.append(coord)
        return thres_coord 

    def preprocess_input(self, image):
        '''
        Change image-data layout from HWC to CHW 
        '''
        h_img = self.input_shape[2]
        w_img = self.input_shape[3]
        
        img = cv2.resize(image, (w_img, h_img))
        img = img.transpose((2, 0, 1))
        img = img.reshape(1, *img.shape)
        
        return img
    

def main(args):
    model=args.model
    device=args.device
    video_file=args.video
    max_people=args.max_people
    threshold=args.threshold
    output_path=args.output_path

    start_model_load_time=time.time()
    pd= PersonDetect(model, device, threshold)
    pd.load_model()
    total_model_load_time = time.time() - start_model_load_time

    queue=Queue()
    
    try:
        queue_param=np.load(args.queue_param)
        for q in queue_param:
            queue.add_queue(q)
    except:
        print("error loading queue param file")

    try:
        cap=cv2.VideoCapture(video_file)
    except FileNotFoundError:
        print("Cannot locate video file: "+ video_file)
    except Exception as e:
        print("Something else went wrong with the video file: ", e)
    
    initial_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    initial_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out_video = cv2.VideoWriter(os.path.join(output_path, 'output_video.mp4'), cv2.VideoWriter_fourcc(*'avc1'), fps, (initial_w, initial_h), True)
    
    counter=0
    start_inference_time=time.time()

    try:
        while cap.isOpened():
            ret, frame=cap.read()
            if not ret:
                break
            counter+=1
            
            coords, image= pd.predict(frame)
            num_people= queue.check_coords(coords)
            print(f"Total People in frame = {len(coords)}")
            print(f"Number of people in queue = {num_people}")
            out_text=""
            y_pixel=25
            
            for k, v in num_people.items():
                out_text += f"No. of People in Queue {k} is {v} "
                if v >= int(max_people):
                    out_text += f" Queue full; Please move to next Queue "
                cv2.putText(image, out_text, (15, y_pixel), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                out_text=""
                y_pixel+=40
            out_video.write(image)
            
        total_time=time.time()-start_inference_time
        total_inference_time=round(total_time, 1)
        fps=counter/total_inference_time

        with open(os.path.join(output_path, 'stats.txt'), 'w') as f:
            f.write(str(total_inference_time)+'\n')
            f.write(str(fps)+'\n')
            f.write(str(total_model_load_time)+'\n')

        cap.release()
        cv2.destroyAllWindows()
    except Exception as e:
        print("Could not run Inference: ", e)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--device', default='CPU')
    parser.add_argument('--video', default=None)
    parser.add_argument('--queue_param', default=None)
    parser.add_argument('--output_path', default='/results')
    parser.add_argument('--max_people', default=2)
    parser.add_argument('--threshold', default=0.60)
    
    args=parser.parse_args()

    main(args)