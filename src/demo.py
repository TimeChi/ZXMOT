from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import os.path as osp
from opts import opts
from tracking_utils.utils import mkdir_if_missing
from track import eval_seq
import time
from tracker.basetrack import BaseTrack


def demo(opt):
    result_root = opt.output_root if opt.output_root != '' else '.'
    mkdir_if_missing(result_root)

    tracklist = os.listdir(opt.data_root)
    for tra in tracklist:
        data_path = os.path.join(opt.data_root, tra)
        result_filename = os.path.join(result_root, tra + '.txt')
        frame_dir = osp.join(result_root, tra)
        eval_seq(opt, data_path, 'mot', result_filename, save_dir=frame_dir, show_image=False)
        BaseTrack.clear_id()



if __name__ == '__main__':
    start = time.time()
    opt = opts().init()
    demo(opt)
    end = time.time()
    print((end-start), ' s')
