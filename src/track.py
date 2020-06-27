from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import numpy as np

from tracker.multitracker import JDETracker
from tracking_utils import visualization as vis

from tracking_utils.utils import mkdir_if_missing
# from opts import opts


def write_results(filename, results, data_type):
    if data_type == 'mot':
        save_format = '{frame},{id},{x1},{y1},{w},{h},1,0\n'
    elif data_type == 'kitti':
        save_format = '{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n'
    else:
        raise ValueError(data_type)

    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids in results:
            if data_type == 'kitti':
                frame_id -= 1
            for tlwh, track_id in zip(tlwhs, track_ids):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h)
                f.write(line)


def eval_seq(opt, data_path, data_type, result_filename, save_dir=None, show_image=True, frame_rate=30):
    tracker = JDETracker(opt)
    # timer = Timer()
    results = []
    frame_id = 0
    frame_nums = 60 #len(os.listdir(data_path))//2
    #np_res = []
    for _ in range(frame_nums):
        frame_id += 1
        dets = np.loadtxt(os.path.join(data_path, str(frame_id) + '.txt'), dtype=np.float32, delimiter=',')

        online_targets = tracker.update(dets)
        online_tlwhs = []
        online_ids = []
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            if tlwh[2] * tlwh[3] > opt.min_box_area and tlwh[2] / tlwh[3] < 1.6:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
                # np_res.append([frame_id,tid,tlwh[0],tlwh[1],tlwh[2],tlwh[3],1,0])

        ## save results
        results.append((frame_id, online_tlwhs, online_ids))

        if show_image or save_dir is not None:
            if save_dir:
                mkdir_if_missing(save_dir)
            img = cv2.imread(os.path.join(data_path, str(frame_id) + '.jpg'))
            online_im = vis.plot_tracking(img, online_tlwhs, online_ids, frame_id=frame_id)
        # if show_image:
        #     cv2.imshow('online_im', online_im)
        if save_dir is not None:
            cv2.imwrite(os.path.join(save_dir, '{:05d}.jpg'.format(frame_id)), online_im)
    # save results
    write_results(result_filename, results, data_type)
    # np.savetxt(result_filename, np.array(np_res), fmt='%d,%d,%0.1f,%0.1f,%0.1f,%0.1f,%0.1f,%d', delimiter=',')

