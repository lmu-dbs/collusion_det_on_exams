import itertools
import time
import editdistance
import pandas as pd
import pm4py
import textdistance

INPUT_PATH = "eip_semestral/logs.csv"
OUTPUT_DIR = "output/"

def import_log_csv(filepath):
    pd.set_option('display.max_columns', None)
    df = pd.read_csv(filepath)
    df = df[["userId", "taskId", "timeStamp", "additionaldata"]]
    df.rename(columns={'userId': 'case:concept:name', 'taskId': 'concept:name', 'timeStamp': 'time:timestamp', 'additionaldata': 'answers'}, inplace=True)
    dataframe = pm4py.format_dataframe(df, case_id='case:concept:name', activity_key='concept:name',
                                       timestamp_key='time:timestamp')
    for i, row in dataframe.iterrows():
        answer = row['answers']
        answer = answer.replace('\"', '').replace('[', '').replace(']', '').replace('}', '').split(":")[1]
        answer_parts = answer.split(",")
        dataframe.at[i, 'answers'] = answer_parts
    return dataframe

# Returns distance based on initial strict conditions
def get_strict_distance(submission_combination, time_threshold):
    if (abs(submission_combination[0][0] - submission_combination[1][0])).total_seconds() < time_threshold \
            and editdistance.eval(submission_combination[0][1], submission_combination[1][1]) == 0:
        return 0
    else:
        return 1

# Returns distance based on whether the submission happened within a time range TIME_THRESHOLD only
def get_time_based_distance(submission_combination, time_threshold):
    if (abs(submission_combination[0][0] - submission_combination[1][0])).total_seconds() < time_threshold:
        return 0
    else:
        return 1

# Return inverse levenshtein distance weighted by time
def get_weighted_distance(submission_combination, time_threshold):
    time_diff = (abs(submission_combination[0][0] - submission_combination[1][0])).total_seconds()
    lev = textdistance.Levenshtein()
    if time_diff > time_threshold:
        return 1
    else:
        #editdist = editdistance.eval(submission_combination[0][1], submission_combination[1][1])
        editdist = lev.normalized_distance(submission_combination[0][1], submission_combination[1][1])
        #return 1-(editdist/time_diff)  # ToDo: normalize, because negative distance could be returned
        return editdist

# Add distance methods for different time ranges

def calculate_task_distance(t1, t2, dist, time_threshold):
    t1_tuple = [tuple(x) for x in t1[['time:timestamp', 'answers']].to_numpy()]
    t2_tuple = [tuple(x) for x in t2[['time:timestamp', 'answers']].to_numpy()]
    two_tuples = [t1_tuple, t2_tuple]
    combinations = [p for p in itertools.product(*two_tuples)]
    distances = []
    for comb in combinations:
        distances.append(eval("get_"+dist+"_distance(comb, time_threshold)"))
    return min(distances)


def sum_norm(min_task_dists):
    return [float(i)/sum(min_task_dists) for i in min_task_dists]

def max_norm(min_task_dists):
    return [float(i) / max(min_task_dists) for i in min_task_dists]

def calculate_user_distance(df, u1, u2, tasks, dist, time_threshold):
    trace_u1 = df.loc[df['case:concept:name'] == u1]
    trace_u2 = df.loc[df['case:concept:name'] == u2]
    min_task_dists = []
    for task in tasks:
        task_trace_u1 = trace_u1.loc[trace_u1['concept:name'] == task]
        task_trace_u2 = trace_u2.loc[trace_u2['concept:name'] == task]
        if not task_trace_u1.empty and not task_trace_u2.empty:
            min_task_dists.append(calculate_task_distance(task_trace_u1, task_trace_u2, dist, time_threshold))
    return sum(min_task_dists)


def compute_distance_matrix(log, dist="strict", time_threshold = "60"):
    counter = 0
    user_dists = dict()
    task_list = log['concept:name'].unique()
    user_list = log['case:concept:name'].unique()
    print(task_list)
    print(user_list)
    combs = (len(user_list)-1)**2
    print(combs)
    for u1 in user_list:
        for u2 in user_list:
            counter += 1
            if counter % 1000 == 0:
                print(f"INFO: Calculating percentage user combination {counter}/{combs}")
            if u1 != u2:
                user_dists[str(u1) + "->" + str(u2)] = calculate_user_distance(log, u1, u2, task_list, dist, time_threshold)
    return user_dists

if __name__ == '__main__':
    DIST_FUNC = 'strict' # "strict", "time_based", or "weighted"
    TIME_THRESHOLD = 60 #In seconds
    SAMPLE_SIZE = 20

    log = import_log_csv(INPUT_PATH)
    #log = log.sample(n=SAMPLE_SIZE)
    start_time = time.time()
    user_dists = compute_distance_matrix(log, DIST_FUNC, TIME_THRESHOLD)
    print(f"Took {time.time() - start_time}")
    print(user_dists)
    OUTPUT_PATH = f"{OUTPUT_DIR}{DIST_FUNC}_distmatrix_{TIME_THRESHOLD}sec.txt"
    with open(OUTPUT_PATH, "w") as f:
        for k, v in user_dists.items():
            f.write(k + ":" + str(v) + "\n")
    print("Saved in "+OUTPUT_PATH)

