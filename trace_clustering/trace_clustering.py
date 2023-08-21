import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import dendrogram, linkage

LINK_CRIT = "single"

INPUT_PATH = "output/weighted_distmatrix_300sec.txt"

def import_preprocessed_data(path):
    dist_matrix = []
    students = []
    with open(path, "r") as f:
        contents = f.readlines()
        for content in contents:
            ref, dist = content.split(":")
            f, t = ref.split("->")
            f = int(f)
            if f not in students:
                students.append(f)
        students.sort()

        for _ in students:
            dist_matrix.append([0] * len(students))

        for content in contents:
            ref, dist = content.split(":")
            dist = float(dist)
            f, t = ref.split("->")
            f = int(f)
            t = int(t)
            i = students.index(f)
            j = students.index(t)
            dist_matrix[i][j] = dist

    return np.array(dist_matrix), np.array(students)


def check_against_gt(l):
    y = sorted([1,2,3,4])

    print(f"Groundtruth has {len(y)} entries")
    print(f"{len(l)} were detected: {sorted(set(l))}")
    return sorted(set(y).intersection(set(l)))

def compute_dendrogram(distmatrix_path, dist, t_thr):
    OUTPUT_PREFIX = distmatrix_path.rsplit(".")[0]
    OUTPUT_PATH = f"{OUTPUT_PREFIX}_dendrogram_{LINK_CRIT}_link.png"

    dist_matrix, students = import_preprocessed_data(distmatrix_path)
    dists = squareform(dist_matrix)
    linkage_matrix = linkage(dists, LINK_CRIT)
    title = f"Dendrogram: {LINK_CRIT} link, {dist} distance, threshold {t_thr} sec"
    plt.figure(figsize=(20, 6), dpi=300)
    fig = dendrogram(linkage_matrix, labels=students, color_threshold=13.0)
    leaves_color_list = fig['leaves_color_list']
    default_color = 'C0'
    leaves_ids = fig['ivl']
    colored_leaves_ids = []
    for idx, item in enumerate(leaves_color_list):
        if item != default_color:
            colored_leaves_ids.append(leaves_ids[idx])
    plt.title(title, fontsize=20)

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH)
    print("INFO: Saved dendrogram in "+OUTPUT_PATH)
    #plt.show()
    in_common = check_against_gt(sorted(colored_leaves_ids))
    print(f"In common {len(in_common)}: {in_common}")


def test_check_against_gt():
    print("Test 1:")
    print(check_against_gt(
        [1,2,3,4]))

    print("Test 2:")
    print(check_against_gt([1,2,3,4]))


if __name__ == '__main__':
    compute_dendrogram(INPUT_PATH, 'weighted', 300)
    #test_check_against_gt()
