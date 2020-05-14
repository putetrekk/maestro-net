import ast

import matplotlib.pyplot as plt
import csv

file = 'output/20200429_120753/sizeinf_epoch50of_50_batch24'

title = 'Transformer'

columns = ['loss', 'val_loss']
# columns = ['accuracy', 'val_accuracy']

with open(file, newline='\n') as csv_file:
    data = list(csv.reader(csv_file, delimiter=','))

# Transpose
data = list(zip(*data))
# Convert to dict with column name as key
data = {x[0]: x[1:] for x in data}
# Parse values to int/float
data = {x: list(map(ast.literal_eval, data[x])) for x in data}

# Plot
for column in columns:
    plt.plot(data['epoch'], data[column])
plt.legend(columns)

plt.title(title)
plt.xlabel('epoch')
plt.show()
