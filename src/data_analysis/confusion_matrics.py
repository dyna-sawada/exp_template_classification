

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


used_temp_id = set(temp_id_1) | set(temp_id_2)
used_temp_id = sorted(list(used_temp_id))
#print(used_temp_id)

cm = confusion_matrix(temp_id_1, temp_id_2, labels=used_temp_id[1:])
print(cm)

cm = pd.DataFrame(
    data=cm,
    index=used_temp_id[1:], 
    columns=used_temp_id[1:]
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
