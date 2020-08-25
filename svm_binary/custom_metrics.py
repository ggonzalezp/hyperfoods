
import numpy as np

def harmonic_mean(pos_metric, neg_metric):
    return ((2.0 * pos_metric * neg_metric) / (pos_metric + neg_metric))


def harmonic_f1(label_vector, pred_vector):
    nac = label_vector == 0
    ac = label_vector == 1

    ac_predictions = pred_vector[ac] == 1
    nac_predictions = pred_vector[nac] == 0

    pos_f1 = np.float32(np.sum(ac_predictions)) / len(ac_predictions)
    neg_f1 = np.float32(np.sum(nac_predictions)) / len(nac_predictions)

    return harmonic_mean(pos_f1, neg_f1)