from utils.export_results import pkl_export, pkl_concat, pkl_import
import numpy as np

def compute_acc(path):
   
    val_data = np.load('competition/val.npy')
    targets = np.array(val_data.item()['labels'])
    results = np.array(pkl_import(path)).reshape(-1, 10)
    preds = np.argmax(results, axis=1)
    print('Accuracy: {:.10f}'.format(np.mean((preds == targets).astype(float))))
