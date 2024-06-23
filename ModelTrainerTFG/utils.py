import csv
import matplotlib.pyplot as plt
import numpy as np

def detect_delimiter(file):
    sample = file.read(1024).decode()
    file.seek(0)  # Reset the file read pointer after reading
    sniffer = csv.Sniffer()
    return sniffer.sniff(sample).delimiter

def plot_confusion_matrix(conf_matrix, ax):
    cax = ax.matshow(conf_matrix, cmap='Blues')
    plt.colorbar(cax, ax=ax)

    for (i, j), val in np.ndenumerate(conf_matrix):
        ax.text(j, i, f'{val}', ha='center', va='center', color='white' if val > np.max(conf_matrix) / 2 else 'black')

    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_xticks(np.arange(conf_matrix.shape[1]))
    ax.set_yticks(np.arange(conf_matrix.shape[0]))
    ax.set_xticklabels(np.arange(conf_matrix.shape[1]))
    ax.set_yticklabels(np.arange(conf_matrix.shape[0]))
    ax.set_title('Confusion Matrix')