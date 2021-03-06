import numpy as np
from sklearn.utils import resample
from Process_data import X_train, y_train, X_test, y_test, train_index, test_index, labels
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import pickle
from sklearn.ensemble import RandomForestClassifier
from scipy import stats
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 



def load_classifier(name):

    if name == "NN":
        # path = os.path.abspath('NN_model.h5')
        # print(path)
        model = load_model('C:/Users/rasmu/OneDrive/Dokumenter/4. semester/Fagprojekt/02466---Project-in-Artificial-Intelligence-and-Data-Science/py/NN_model.h5')

    elif name == "RF":
        # path = os.path.abspath('RF.h5')
        model = pickle.load(open('C:/Users/rasmu/OneDrive/Dokumenter/4. semester/Fagprojekt/02466---Project-in-Artificial-Intelligence-and-Data-Science/py/RF.sav', 'rb'))
    else:
        print("Wrong model name")
        return 

    return model

def permutation(n_perm, name, plots = False):
    model = load_classifier(name)

    X_perm = X_test
    n_attr = X_test.shape[1]

    accs = np.zeros([n_perm,n_attr])
    losses = np.zeros([n_perm,n_attr])

    for idx in range(n_attr):
        for trial in range(n_perm):
            X_perm = X_test
            X_perm[:,idx] = resample(X_perm[:,idx] ,replace = False)
            if name == "NN":
                perm_loss, perm_accuracy = model.evaluate(X_perm, y_test, verbose = 0)
            elif name == "RF":
                perm_accuracy = model.score(X_perm,y_test)
            else:
                print("Wrong name")
                return

            accs[trial,idx] = perm_accuracy 
            #losses[trial,idx] = perm_loss
                     
    #loss_mean = np.mean(losses,axis=0) 
    accs_mean = np.mean(accs, axis = 0)

    #error_loss = np.std(losses, axis = 0)/np.sqrt(n_perm)
    error_accs = np.std(accs, axis = 0 )/np.sqrt(n_perm)

    y_pos = np.arange(n_attr)

#    plt.subplot(211)

#    plt.barh(y_pos, loss_mean, xerr = error_loss)
#    plt.title("Loss")
#    plt.yticks(y_pos,labels)

#    plt.subplot(212)
    if plots == True:
        plt.barh(y_pos,accs_mean)
        plt.title(name + " - " + "Accuracy")
        plt.yticks(y_pos,labels)
        plt.show()

    return accs

def permutation_test(n_perm, name):

    accs = permutation(n_perm, name)
    
    p_values = []

    for idx in range(accs.shape[1]):
        p = stats.ttest_1samp(accs[:,idx],0.81)[1]
        p_values.append(p)

    return p_values, accs



p_values, accs = permutation_test(2, "NN")


p_path = 'C:/Users/rasmu/OneDrive/Dokumenter/4. semester/Fagprojekt/02466---Project-in-Artificial-Intelligence-and-Data-Science/py/p_values_29.npy'
accs_path = 'C:/Users/rasmu/OneDrive/Dokumenter/4. semester/Fagprojekt/02466---Project-in-Artificial-Intelligence-and-Data-Science/py/accs_29.npy'

np.save(p_path, p_values)
np.save(accs_path, accs)

p = np.load('C:/Users/rasmu/OneDrive/Dokumenter/4. semester/Fagprojekt/02466---Project-in-Artificial-Intelligence-and-Data-Science/py/p_values_29.npy')
a = np.load('C:/Users/rasmu/OneDrive/Dokumenter/4. semester/Fagprojekt/02466---Project-in-Artificial-Intelligence-and-Data-Science/py/accs_29.npy')

print(p, "\n", a)