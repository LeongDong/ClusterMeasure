import numpy as np

def silhouette_coefficient(img, mask):

    classNum = np.unique(mask).shape[0]
    sc = []
    for i in range(classNum):
        index = (mask == i)
        i_class = img[index]
        for sample in i_class:
            intra_class_diff = (sample - i_class) * (sample - i_class)
            intra_class_diff = np.sqrt(np.sum(intra_class_diff)) / (len(intra_class_diff) - 1)

            nearest_class_num = None
            min_inter_class_center = np.inf
            for j in range(classNum):
                if(i == j):
                    continue
                classCenter = np.mean(img[mask == j])
                inter_class_diff = (sample - classCenter) * (sample - classCenter)
                if(inter_class_diff < min_inter_class_center):
                    min_inter_class_center = inter_class_diff
                    nearest_class_num = j

            nearest_class = img[mask == nearest_class_num]
            min_inter_class_diff = (sample - nearest_class) * (sample - nearest_class)
            min_inter_class_diff = np.sqrt(np.sum(min_inter_class_diff)) / len(min_inter_class_diff)
            sc.append((min_inter_class_diff - intra_class_diff) / np.max([intra_class_diff, min_inter_class_diff]))

    return np.mean(sc)