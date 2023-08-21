import pm4py
import time

from trace_clustering.preprocess_trace_clustering import import_log_csv, compute_distance_matrix
from trace_clustering.trace_clustering import import_preprocessed_data, compute_dendrogram

DIST_FUNCS = ['strict','time_based','weighted'] # "strict", "time_based", "weighted"
TIME_THRESHOLDS = [60, 120, 180, 240, 300] #In seconds
SAMPLE_SIZE = 20

INPUT_PATH = "eip_semestral/logs.csv"
OUTPUT_DIR = "output/"

def preprocess_once(dist, time_threshold):
    distmatrix_path = f"{OUTPUT_DIR}{dist}_distmatrix_{time_threshold}sec.txt"

    log = import_log_csv(INPUT_PATH)
    log = log.sample(n=SAMPLE_SIZE)
    start_time = time.time()
    user_dists = compute_distance_matrix(log, dist, time_threshold)
    print(f"INFO: Distance matrix computation took {time.time() - start_time}")
    print(user_dists)
    with open(distmatrix_path, "w") as f:
        for k, v in user_dists.items():
            f.write(k + ":" + str(v) + "\n")
    print("INFO: Saved in "+distmatrix_path)

def run_once(dist, t_thr):
    distmatrix_path = f"{OUTPUT_DIR}{dist}_distmatrix_{t_thr}sec.txt"

    preprocess_once(dist, t_thr)
    compute_dendrogram(distmatrix_path, dist, t_thr)

if __name__ == '__main__':
    for dist in DIST_FUNCS:
        for t_thr in TIME_THRESHOLDS:
            run_once(dist, t_thr)
