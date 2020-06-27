from tracking_utils.utils import *

from tracking_utils.kalman_filter import KalmanFilter

from tracker import matching
from .basetrack import BaseTrack, TrackState


class STrack(BaseTrack):
    shared_kalman = KalmanFilter()
    def __init__(self, tlwh, score):#, temp_feat, buffer_size=30):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id


    def update(self, new_track, frame_id, update_feature=True):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class JDETracker(object):
    def __init__(self, opt):
        self.opt = opt
        self.tracked_stracks = []  # type: list[STrack]
        self.frame_id = 0
        self.kalman_filter = KalmanFilter()

    def nms(self,dets, max_bbox_overlap):
        if len(dets) == 0:
            return []
        elif len(dets) == 1:
            return [0]

        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 0] + dets[:, 2]
        y2 = dets[:, 1] + dets[:, 3]

        scores = dets[:, 4]  # bbox打分

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        # 打分从大到小排列，取index
        order = scores.argsort()[::-1]
        # keep为最后保留的边框
        keep = []
        # print("order:",order)
        while order.size > 0:
            # order[0]是当前分数最大的窗口，肯定保留
            i = order[0]
            keep.append(i)
            # 计算窗口i与其他所有窗口的交叠部分的面积
            xx1 = [max(x1[i], x) for x in x1[order[1:]]]
            xx2 = [min(x2[i], x) for x in x2[order[1:]]]
            yy1 = [max(y1[i], y) for y in y1[order[1:]]]
            yy2 = [min(y2[i], y) for y in y2[order[1:]]]

            w = np.array([max(0.0, xx2[i] - xx1[i] + 1) for i in range(len(xx1))])
            h = np.array([max(0.0, yy2[i] - yy1[i] + 1) for i in range(len(yy1))])
            inter = w * h
            # 交/并得到iou值
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            # inds为所有与窗口i的iou值小于threshold值的窗口的index，其他窗口此次都被窗口i吸收
            inds = np.where(ovr <= max_bbox_overlap)[0]
            # order里面只保留与窗口i交叠面积小于threshold的那些窗口，由于ovr长度比order长度少1(不包含i)，所以inds+1对应到保留的窗口

            order = order[inds + 1]  # inds 的第一个索引对应order的第二个索引

        return keep

    def update(self, dets):
        self.frame_id += 1
        activated_starcks = []

        remain_inds = dets[:, -1] > self.opt.det_th
        dets = dets[remain_inds]
        keep = self.nms(dets, 0.7)
        dets = dets[keep]

        if len(dets) > 0:
            '''Detections'''
            detections = [STrack(tlbrs[:4], tlbrs[4]) for
                          tlbrs in dets[:, :5]]
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        tracked_stracks = []  # type: list[STrack]
        Prio_match = False
        if len(self.tracked_stracks) > 10 and self.opt.tracklet_score:
            Prio_match = True
            max_len = max([tra.tracklet_len for tra in self.tracked_stracks])+1
        scores = []
        for track in self.tracked_stracks:
            tracked_stracks.append(track)
            if Prio_match:
                scores.append(track.tracklet_len/max_len + track.score)# 轨迹评分函数

        strack_pool = tracked_stracks
        if Prio_match:
            order = np.array(scores).argsort()[::-1]
            pri_strack_pool = [tracked_stracks[i] for i in order[:order.size // 2]]
            dists = matching.iou_distance(pri_strack_pool, detections)
            matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.5)

            for itracked, idet in matches:
                track = pri_strack_pool[itracked]
                if track.state == TrackState.Tracked:
                    track.update(detections[idet], self.frame_id)
                    activated_starcks.append(track)
            strack_pool = [tracked_stracks[i] for i in order[order.size // 2:]]+[pri_strack_pool[i] for i in u_track if pri_strack_pool[i].state == TrackState.Tracked]
            detections = [detections[i] for i in u_detection]
        ''' Step 2: First association, with kf'''
        STrack.multi_predict(strack_pool)
        dists = np.zeros((len(strack_pool), len(detections)), dtype=np.float)
        dists = matching.gate_cost_matrix(self.kalman_filter, dists, strack_pool, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.7)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
        ''' Step 3: Second association, with IOU'''
        detections = [detections[i] for i in u_detection]
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.5)

        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_removed()


        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < 0.85:#self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        return output_stracks


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb