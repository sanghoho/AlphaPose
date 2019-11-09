import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as transforms

import torch.nn as nn
import torch.utils.data
import numpy as np

from dataloader import ImageLoader, DetectionLoader, DetectionProcessor, DataWriter, Mscoco
from fn import getTime
from pPose_nms import pose_nms, write_json

from yolo.util import write_results, dynamic_write_results
from SPPE.src.main_fast_inference import *

import os
import glob
import sys
from tqdm import tqdm
import time

class PoseEstimator:
    def __init__(self, input_path, output_path, is_single=True):
        super(PoseEstimator, self).__init__()
        torch.multiprocessing.set_start_method('spawn', force=True)
        self.inputpath = input_path
        self.outputpath = output_path
        self.processing_option = is_single

        if not os.path.exists(self.outputpath):
            os.mkdir(self.outputpath)
        
        if not self.processing_option:
            torch.multiprocessing.set_start_method('forkserver', force=True)
            torch.multiprocessing.set_sharing_strategy('file_system')
        else:
            torch.multiprocessing.set_start_method('spawn', force=True)

        torch.cuda.empty_cache()


    def run(self):

        # Load input images
        data_loader = ImageLoader(self.inputpath, batchSize=1, format='yolo').start()

        print('Loading YOLO model..')
        sys.stdout.flush()

        # Load detection loader
        det_loader = DetectionLoader(data_loader, batchSize=1).start()
        det_processor = DetectionProcessor(det_loader).start()
        
        # Load pose model
        pose_dataset = Mscoco()
        pose_model = InferenNet_fast(4 * 1 + 1, pose_dataset)
        pose_model.cuda()
        pose_model.eval()

        runtime_profile = {
            'dt': [],
            'pt': [],
            'pn': []
        }

        # Init data writer
        self.writer = DataWriter(False).start()

        data_len = data_loader.length()
        im_names_desc = tqdm(range(data_len))

        batchSize = 80
        for i in im_names_desc:
            start_time = getTime()
            with torch.no_grad():
                (inps, orig_img, im_name, boxes, scores, pt1, pt2) = det_processor.read()
                if boxes is None or boxes.nelement() == 0:
                    self.writer.save(None, None, None, None, None, orig_img, im_name.split('/')[-1])
                    continue

                ckpt_time, det_time = getTime(start_time)
                runtime_profile['dt'].append(det_time)
                # Pose Estimation
                
                datalen = inps.size(0)
                leftover = 0
                if (datalen) % batchSize:
                    leftover = 1
                num_batches = datalen // batchSize + leftover
                hm = []
                for j in range(num_batches):
                    inps_j = inps[j*batchSize:min((j +  1)*batchSize, datalen)].cuda()
                    hm_j = pose_model(inps_j)
                    hm.append(hm_j)
                hm = torch.cat(hm)
                ckpt_time, pose_time = getTime(ckpt_time)
                runtime_profile['pt'].append(pose_time)
                hm = hm.cpu()
                self.writer.save(boxes, scores, hm, pt1, pt2, orig_img, im_name.split('/')[-1])

                ckpt_time, post_time = getTime(ckpt_time)
                runtime_profile['pn'].append(post_time)
            torch.cuda.empty_cache()
            


        print('===========================> Finish Model Running.')
    
        while(self.writer.running()):
            pass
        self.writer.stop()

    def save_result(self):    
        final_result = self.writer.results()
        return write_json(final_result, self.outputpath)