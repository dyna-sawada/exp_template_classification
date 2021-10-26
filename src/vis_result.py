

import matplotlib.pyplot as plt
import json
import numpy as np



result_file = './out_test/iter_0/train_log_fold_0.json'

result_data = json.load(open(result_file))

train_losses = result_data['train_losses']
valid_losses = result_data['val_losses']
coverages = result_data['coverage_error']


print(train_losses, valid_losses, coverages)

epoch = [i for i, _ in enumerate(train_losses)]
print(epoch)


plt.plot(epoch, coverages)
plt.show()


