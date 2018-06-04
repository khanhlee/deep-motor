import numpy

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# Create first network with Keras
from keras.utils import np_utils
from keras.models import model_from_json
print(__doc__)
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from scipy import interp


#define params
tst_file = 'dataset/multiclass.indep.csv'
json_model = 'final_model.json'
h5_model = 'final_model.h5'
#output = sys.argv[4]

nb_classes = 4
nb_kernels = 3
nb_pools = 2
window_sizes = 19

# load testing dataset
dataset1 = numpy.loadtxt(tst_file, delimiter=",")
# split into input (X) and output (Y) variables
X1 = dataset1[:,0:20*20].reshape(len(dataset1),1,20,20)
Y1 = dataset1[:,20*20]

Y1 = np_utils.to_categorical(Y1,nb_classes)

# load json and create model
json_file = open(json_model, 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(h5_model)
print("Loaded model from disk")
predictions = loaded_model.predict(X1)

#print(predictions)
#f = open(output,'w')
#output = np_utils.categorical_probas_to_classes(predictions)
#for x in output:
#    f.write(str(x) + '\n')
#f.close()

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(4):
    fpr[i], tpr[i], _ = roc_curve(Y1[:, i], predictions[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(Y1.ravel(), predictions.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


##############################################################################
# Plot of a ROC curve for a specific class
plt.figure()
lw = 1
plt.plot(fpr[1], tpr[1], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[1])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

# Compute macro-average ROC curve and ROC area
# First aggregate all false positive rates
all_fpr = numpy.unique(numpy.concatenate([fpr[i] for i in range(4)]))

# Then interpolate all ROC curves at this points
mean_tpr = numpy.zeros_like(all_fpr)
for i in range(4):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= 4

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure(figsize=(10,7))
#plt.plot(fpr["micro"], tpr["micro"],
#         label='micro-average ROC curve (AUC = {0:0.2f})'
#               ''.format(roc_auc["micro"]),
#         color='deeppink', linestyle=':', linewidth=4)
#
#plt.plot(fpr["macro"], tpr["macro"],
#         label='macro-average ROC curve (AUC = {0:0.2f})'
#               ''.format(roc_auc["macro"]),
#         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'gray'])
for i, color in zip(range(4), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw, linewidth=4,
             label='ROC curve of class {0} (AUC = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=15)
plt.ylabel('True Positive Rate', fontsize=15)
#plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right", prop={'size': 15})
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()
plt.savefig('roc_final.svg')
