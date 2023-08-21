import itertools
from datetime import datetime

import numpy as np
import pandas as pd
import scipy
from matplotlib import pyplot as plt
from pyclustering.cluster.optics import optics
from scipy.signal import find_peaks


def import_participations_csv(filepath):
    df = pd.read_csv(filepath)
    cheating = []
    for index, row in df.iterrows():
        if "Unterschleif" in str(row['commentfornotpassed']):
            cheating.append(int(row['uid']))
    cheating.sort()
    return cheating


def import_log_csv(filepath):
    df = pd.read_csv(filepath)

    traces = dict()

    for index, row in df.iterrows():
        userId = int(row['userId'])
        taskId = int(row['taskId'])
        timestamp = datetime.strptime(row['timeStamp'], "%Y-%m-%d %H:%M:%S")

        if userId in traces:
            traces[userId].append((taskId, timestamp))
        else:
            traces[userId] = [(taskId, timestamp)]

    # print(traces[317])
    temp = []
    for t in traces:
        temp.append(len(traces[t]))
    print(np.median(temp))
    return traces


class Toad:

    def __init__(self, log, minpts, eps, prominence):
        self.log = log
        self.minpts = minpts
        self.eps = eps
        self.prominence = prominence

    def apply_toad(self):
        d = {}
        Z = {}
        remap_tids = []

        all_rels = []
        for tid, trace in self.log.items():
            trace_act = [e for e, t in trace]

            z = {}
            for (e1, e2) in list(itertools.combinations(trace, 2)):

                a1 = e1[0]
                t1 = e1[1]
                a2 = e2[0]
                t2 = e2[1]

                rel = str(a1) + "->" + str(a2)
                diff = (t2 - t1).total_seconds()
                if rel in d:
                    d[rel].append(diff)
                else:
                    d[rel] = [diff]
                z[rel] = diff
            Z[tid] = z

        avg = {}
        std = {}
        ext = {}

        for rel, values in d.items():
            avg[rel] = np.mean(values)
            std[rel] = np.std(values)
            ext[rel] = np.max(np.abs(values))

        rels = list(d.keys())

        test_time = []

        # standardizing
        Zstd = {}
        for tid, trace in Z.items():
            vstd = {}
            for rel, value in trace.items():
                vstd[rel] = (value - avg[rel])
                if std[rel] == 0:
                    vstd[rel] = 0
                else:
                    vstd[rel] /= std[rel]
            Zstd[tid] = vstd

        tids = Z.keys()

        Zvectors = []

        for tid in tids:

            temp = Zstd[tid]
            dummy = []
            for rel in rels:
                if rel in temp:
                    dummy.append(temp[rel])
                else:
                    dummy.append(0.0)
            Zvectors.append(dummy)
            remap_tids.append(tid)

        print("Number of traces: ", len(Zvectors))
        print("Number of relations: ", len(rels))

        optics_instance = optics(Zvectors, self.eps, self.minpts)

        optics_instance.process()

        clusters = optics_instance.get_clusters()
        noise = optics_instance.get_noise()

        reach = pd.Series(optics_instance.get_ordering())
        left = max(reach)

        reach_smoothed = scipy.signal.savgol_filter(reach, 5, 3)

        if len(reach) % 2 == 0:
            l = len(reach) - 1
        else:
            l = len(reach) - 2

        yhat = scipy.signal.savgol_filter(reach, l, 3)

        coord1 = [np.array((x, reach_smoothed[x])) for x in range(len(reach_smoothed))]
        coord2 = [np.array((x, yhat[x])) for x in range(len(yhat))]

        diffy = (yhat - reach_smoothed) * yhat

        yhatclip = np.clip(diffy, 0, np.max(diffy))

        peaks, properties = find_peaks(yhatclip, prominence=self.prominence, width=self.minpts / 2.0)

        cc = {}
        cluster_objects_ids = {}
        entropy_all = {}

        outliers_list = []

        clusters = [val for sublist in clusters for val in sublist]

        for k in range(len(peaks)):
            left = np.ceil(properties["left_ips"][k])
            right = np.floor(properties["right_ips"][k])
            cc[k] = [clusters[l] for l in range(int(left), int(right))]

            outliers_list.extend(cc[k])
            cluster = [Zvectors[l] for l in cc[k]]

            cluster_objects_ids[k] = [remap_tids[o] for o in cc[k]]

            print(cluster_objects_ids)

            entropies = []
            for i in range(len(rels)):
                rel = rels[i]
                e = np.std([v[i] for v in cluster])
                mean = np.mean([v[i] for v in cluster])

                entropies.append((e, rel, mean))

            entropy_all[k] = entropies

        fig, ax1 = plt.subplots(figsize=(8, 2))
        # fig = plt.figure(figsize=(8, 4), dpi=dpi)

        plt.plot(reach_smoothed)
        plt.plot(yhat, color='red')

        if len(peaks) == 1:
            xs = np.arange(int(properties["left_ips"]), int(properties["right_ips"]), 1)
            ax1.fill_between(xs, reach_smoothed[int(properties["left_ips"]):int(properties["right_ips"])],
                             yhat[int(properties["left_ips"]):int(properties["right_ips"])], color='00', alpha=0.3)
        elif len(peaks) > 1:
            for i in range(len(peaks)):
                xs = np.arange(int(properties["left_ips"][i]), int(properties["right_ips"][i]), 1)
                ax1.fill_between(xs, reach_smoothed[int(properties["left_ips"][i]):int(properties["right_ips"][i])],
                                 yhat[int(properties["left_ips"][i]):int(properties["right_ips"][i])], color='00',
                                 alpha=0.3)
            # plt.vlines(x=peaks, ymin=reach_smoothed[peaks], ymax = yhat[peaks], color = "C1")
        # plt.hlines(y=reach_smoothed[peaks], xmin=properties["left_ips"], xmax=properties["right_ips"], color = "C1")
        plt.title('TOAD plot')
        plt.ylabel('Reachability Distance')
        plt.xlabel('Traces')
        plt.show()
        return cluster_objects_ids


if __name__ == '__main__':
    # log = import_log_csv('eip_semestral/logs.csv')
    log = import_log_csv('eip_semestral/logs_last.csv')
    # print([x for x, y in log[60]])
    # 50, 30, 0.3

    toad = Toad(log, 3, 20, 0.7)
    candidates = toad.apply_toad()

    gt = import_participations_csv('eip_semestral/participations.csv')
    print(gt)
    print(len(gt))
    for cluster, c in candidates.items():
        print(f"Checking cluster {c}, {len(c)}")
        print(set(gt).intersection(set(c)))

##
y = [2,3,4,5]
a = [1,23,4,6]

print(len(y), len(a))
print(set(y).intersection(set(a)))
