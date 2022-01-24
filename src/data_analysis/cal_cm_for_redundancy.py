

import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


DATA_DIR = './work/for_redundancy_samples.xlsx'

df = pd.read_excel(DATA_DIR)
df = df.fillna(0)


df['temp_id_1'] = df['temp_id_1'].astype(int)
df['temp_id_2'] = df['temp_id_2'].astype(int)

temp_id_1 = df['temp_id_1'].to_list()
temp_id_2 = df['temp_id_2'].to_list()


valid_count = 0
miss_count = 0
for t1, t2 in zip(temp_id_1, temp_id_2):
    if t1 == 0 or t2 == 0:
        valid_count += 1
    else:
        if t1 != t2:
            miss_count += 1

print(
    'N_all\t{}\nN_valid\t{}\nN_miss\t{}'.format(
        len(temp_id_1),
        valid_count,
        miss_count
    )
)

cm_labels = [10, 20, 30, 40, 50, 60, 70, 80, 100, 110, 121, 148, 150, 160, 90, 131, 140, 170, 180, 190, 200, 211, 220, 230]
tm_labels = [
    'CA1', 'CA2', 'CA3', 'CA4',
    'VAL1', 'VAL2', 'VAL3', 'VAL4',
    'CLS1', 'CLS2', 'PR1',
    'EX1', 'EX2', 'EX3', 'CMP1', 'CMP2',
    'LR1', 'CLR1', 'CLR2',
    'GR1', 'GR2', 'GR3', 'GS1', 'GS2'    
]

used_temp_id = set(temp_id_1) | set(temp_id_2)
used_temp_id = sorted(list(used_temp_id))


cm = confusion_matrix(temp_id_1, temp_id_2, labels=cm_labels)
print('--- confusion matrics ---')
print(cm)

cm = pd.DataFrame(
    data=cm,
    index=tm_labels, 
    columns=tm_labels
)

sns.heatmap(
    cm,
    square=True,
    cbar=True,
    annot=True,
    cmap='Blues'
)

plt.xlabel("Annotator B", fontsize=13)
plt.ylabel("Annotator A", fontsize=13)
plt.xticks(rotation=0)
plt.show()
#plt.savefig('./trash/data/confusion_matrix_for_redundancy.png')
