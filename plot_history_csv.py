import ast
import matplotlib.pyplot as plt
import csv
import os
from os.path import isfile, join

# USE (DIR_NAME, NETWORK_LABEL)
DIRS_TO_PLOT = [
    ("20200526_095935_lstm_all", "LSTM 100 units"),
    ("20200526_114934_lstm_all_512", "LSTM 512 units"),
]

COLUMNS_TO_PLOT = [
    'loss',
    'val_loss',
    #'accuracy',
    #'val_accuracy',
]

PLOT_TITLE = "LSTM ON ALL 4 256 530 560 SAMPLES"

LINE_STYLES = [
    '-',
    '--',
    #'-.',
    #':',
]

COLORS = [
    #'b',
    #'r',
    #'y',
    #'c',
    #'m',
    #'g',
    '0.75',
    '0.5',
    '0.25',
    '0',
]

# will be set automatically
column_labels = []


def next_line_style():
    style = LINE_STYLES.pop(0)
    LINE_STYLES.append(style)
    return style


def next_color():
    style = COLORS.pop(0)
    COLORS.append(style)
    return style


def plot_csv(file:str, network_label:str):

    with open(file, newline='\n') as csv_file:
        data = list(csv.reader(csv_file, delimiter=','))

    # Transpose
    data = list(zip(*data))
    # Convert to dict with column name as key
    data = {x[0]: x[1:] for x in data}
    # Parse values to int/float
    data = {x: list(map(ast.literal_eval, data[x])) for x in data}

    color = next_color()

    # Plot
    for column in COLUMNS_TO_PLOT:
        plt.plot(data['epoch'], data[column], linestyle=next_line_style(), lw=2, c=color)
        column_labels.append(f'{network_label}: {column}')
    plt.legend(column_labels)

    plt.title(PLOT_TITLE)
    plt.xlabel('epoch')


if __name__ == '__main__':
    os.chdir("output")

    dirs_in_folder = os.listdir()

    for dir in DIRS_TO_PLOT:
        os.chdir(dir[0])

        files = [f for f in os.listdir() if isfile(f) and f.split('.')[1] == 'csv']
        lastcsv = sorted(files, reverse=True)[0]

        plot_csv(lastcsv, network_label=dir[1])

        os.chdir('..')

    plt.show()
