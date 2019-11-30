import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as transforms

import torch.nn as nn
import torch.utils.data
import numpy as np

from ..utils.dataloader import ImageLoader, DetectionLoader, DetectionProcessor, DataWriter, Mscoco
from ..utils.fn import getTime
from ..utils.pPose_nms import pose_nms, write_json

from ..yolo.util import write_results, dynamic_write_results
from ..SPPE.src.main_fast_inference import *

import os
import glob
import sys
from tqdm import tqdm
import time

class HyperParameter:
    def __init__(self, cuda, param_dict):
        self.cuda = cuda

        if self.cuda and not torch.cuda.is_available():
            print("ERROR: cuda is not available, try running on CPU")
            sys.exit(1)

        self.set_parameter(param_dict)
        
    def set_parameter(self, param_dict):
        # Required
        self.input_path = param_dict["input_path"]
        self.output_path = param_dict["output_path"]

        # Optional
        self.single_process = param_dict.get("single_process", 1)
        self.det_batch = param_dict.get("det_batch", 1)
        self.fast_inference = param_dict.get("fast_inference", 1)
        self.pose_batch = param_dict.get("pose_batch", 80)
        self.profile = param_dict.get("profile", 0)

        self.save_img = param_dict.get("save_img", 0)
        self.save_video = param_dict.get("save_video", 0) 
        self.vis_fast = param_dict.get("vis_fast", 0)
        



def estimate(param):

    if not param.single_process:
        torch.multiprocessing.set_start_method('forkserver', force=True)
        torch.multiprocessing.set_sharing_strategy('file_system')

    if not os.path.exists(param.output_path):
        os.mkdir(param.output_path)


    # torch.cuda.empty_cache()
    device = torch.device("cuda" if param.cuda else "cpu")

    # Load input images
    data_loader = ImageLoader(param.input_path, batchSize=1, format='yolo').start()

    print('Loading YOLO model..')
    sys.stdout.flush()


    # Load detection loader
    det_loader = DetectionLoader(data_loader, param.cuda, batchSize=1).start()
    det_processor = DetectionProcessor(det_loader).start()
    
    # Load pose model
    pose_dataset = Mscoco()
    if param.fast_inference:
        pose_model = InferenNet_fast(4 * 1 + 1, pose_dataset, device)
    else:
        pose_model = InferenNet(4 * 1 + 1, pose_dataset, device)
    pose_model.to(device)
    pose_model.eval()

    print("eval finish")

    runtime_profile = {
        'dt': [],
        'pt': [],
        'pn': []
    }

    # Init data writer
    writer = DataWriter(param.save_video).start()

    data_len = data_loader.length()
    im_names_desc = tqdm(range(data_len))

    batchSize = param.pose_batch
    for i in im_names_desc:
        start_time = getTime()
        with torch.no_grad():
            (inps, orig_img, im_name, boxes, scores, pt1, pt2) = det_processor.read()
            if boxes is None or boxes.nelement() == 0:
                writer.save(None, None, None, None, None, orig_img, im_name.split('/')[-1])
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
                inps_j = inps[j*batchSize:min((j +  1)*batchSize, datalen)].to(device)
                hm_j = pose_model(inps_j)
                hm.append(hm_j)
            hm = torch.cat(hm)
            ckpt_time, pose_time = getTime(ckpt_time)
            runtime_profile['pt'].append(pose_time)
            hm = hm.cpu()
            self.writer.save(boxes, scores, hm, pt1, pt2, orig_img, im_name.split('/')[-1])

            ckpt_time, post_time = getTime(ckpt_time)
            runtime_profile['pn'].append(post_time)

        if param.profile:
            # TQDM
            im_names_desc.set_description(
            'det time: {dt:.3f} | pose time: {pt:.2f} | post processing: {pn:.4f}'.format(
                dt=np.mean(runtime_profile['dt']), pt=np.mean(runtime_profile['pt']), pn=np.mean(runtime_profile['pn']))
            )
        # torch.cuda.empty_cache()
        
    print('===========================> Finish Model Running.')

    if (param.save_img or param.save_video) and not param.vis_fast:
        print('===========================> Rendering remaining images in the queue...')
        print('===========================> If this step takes too long, you can enable the --vis_fast flag to use fast rendering (real-time).')
    while(writer.running()):
        pass
    writer.stop()

    final_result = writer.results()
    write_json(final_result, param.output_path)

    
