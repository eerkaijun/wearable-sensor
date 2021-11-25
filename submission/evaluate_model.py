import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from seaborn import heatmap
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, accuracy_score, recall_score

def get_args():
    """
    Returns a namedtuple with arguments extracted from the command line.
    :return: A namedtuple with arguments
    """
    parser = argparse.ArgumentParser(
        description='Welcome to the PDIoT Inference script')

    parser.add_argument('--model_path', type=str, default='./Models/model_conv_lstm.tflite', help='Path to trained model or folder containing the models')
    parser.add_argument('--test_data_path', type=str, default=' ', help='Path to Respeck test data')

    args = parser.parse_args()
    return args

def data_process_pipeline(base_df, window_size, step_size, class_labels):
    # Let's remove the general movement recordings
    base_df = base_df[base_df['activity_type'] != 'Movement']

    window_number = 0
    all_overlapping_windows = []
    for rid, group in base_df.groupby("recording_id"):
        large_enough_windows = [window for window in group.rolling(window=window_size, min_periods=window_size) if len(window) == window_size]
        if len(large_enough_windows) == 0:
            continue
        overlapping_windows = large_enough_windows[::step_size]
        # then we will append a window ID to each window
        for window in overlapping_windows:
            window.loc[:, 'window_id'] = window_number
            window_number += 1
        all_overlapping_windows.append(pd.concat(overlapping_windows).reset_index(drop=True))

    final_sliding_windows = pd.concat(all_overlapping_windows).reset_index(drop=True)
    columns_of_interest = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']
    X = []
    y = []
    #y_original = []
    for window_id, group in final_sliding_windows.groupby('window_id'):
        shape = group[columns_of_interest].values.shape
        X.append(group[columns_of_interest].values)
        y.append(class_labels[group["activity_type"].values[0]])
        #y_original.append(class_labels_original[group["activity_type"].values[0]])

    X = np.asarray(X)
    y = np.asarray(y)
    #y_original = np.asarray(y_original)

    n_steps, n_length, n_features = 2, 25, 6
    X = X.reshape((X.shape[0], n_steps, 1, n_length, n_features))

    return X, y

def plot_confusion_matrix(cm, filename, classes=None, title='Confusion matrix'):
    """Plots a confusion matrix."""
    if classes is not None:
        heatmap(cm, xticklabels=classes, yticklabels=classes, vmin=0., vmax=1., annot=True)
    else:
        heatmap(cm, vmin=0., vmax=1.)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(filename)

def evaluate_and_visualise_subset(model_path, X, y, filename, class_names):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    y_pred = []
    for item in X:
        interpreter.set_tensor(input_details[0]['index'], np.float32(np.array([item])))
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        y_pred.append(np.argmax(output_data[0]))
    y_pred = np.array(y_pred)

    print("*" * 70)
    print("Classification report for subset of activities")
    print("*" * 70)
    print(classification_report(y, y_pred))

    # plot confusion matrix
    cm = confusion_matrix(y, y_pred)
    cm_norm = cm/cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.around(cm_norm, 2)

    plt.figure()
    plot_confusion_matrix(cm_norm, filename, classes=['Falling', 'Sitting/Standing', 'Lying', 'Walking', 'Running'],
    title='Confusion matrix for subset of activities')

    components = classification_report(y, y_pred, output_dict=True)
    for key in components:
        if key not in class_names:
            break
        precision = str(round(components[key]['precision'],2))
        f_score = str(round(components[key]['f1-score'],2))
        recall = str(round(components[key]['recall'],2))
        classname = class_names[key]
        key = int(key)
        print(classname+' ........... Accuracy: '+str(cm_norm[key][key]) + ', Precision: ' + precision+ ', Recall: ' + recall + ', F-Score: ' + f_score)

def main():
    args = get_args()
    base_df = pd.read_csv(args.test_data_path)
    class_labels = {
        'Climbing stairs': 3,
        'Descending stairs': 3,
        'Desk work': 1,
        'Falling on knees': 0,
        'Falling on the back': 0,
        'Falling on the left': 0,
        'Falling on the right': 0,
        'Lying down left': 2,
        'Lying down on back': 2,
        'Lying down on stomach': 2,
        'Lying down right': 2,
        'Running': 4,
        'Sitting': 1,
        'Sitting bent backward': 1,
        'Sitting bent forward': 1,
        'Standing': 1,
        'Walking at normal speed': 3
    }
    class_names = {
        '0': 'Falling',
        '1': 'Sitting/Standing',
        '2': 'Lying',
        '3': 'Walking',
        '4': 'Running'
    }
    X, y = data_process_pipeline(base_df, 50, 25, class_labels)
    evaluate_and_visualise_subset(args.model_path, X, y, 'confusion_matrix.png', class_names)


if __name__ == "__main__":
    main()
