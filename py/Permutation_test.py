import numpy as np
from sklearn.utils import resample
from Process_data import X_train, y_train, X_test, y_test, train_index, test_index
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model


def Permutation(n_perm, model_type):
    X_perm = X_test

    n_perm = 10
    n_attr = 29

    accs = np.zeros([n_perm,n_attr])
    losses = np.zeros([n_perm,n_attr])

    for idx in range(n_attr):
        for trial in range(n_perm):
            X_perm = X_test
            X_perm[:,idx] = resample(X_perm[:,idx] ,replace = False)
            if model_type == "NN":
                model = load_model("C:/Users/rasmu/OneDrive/Dokumenter/4. semester/Fagprojekt/02466---Project-in-Artificial-Intelligence-and-Data-Science/py/First.h5")
                perm_loss, perm_accuracy = model.evaluate(X_perm, y_test)
            elif model_type == "RF":
                predictions = model.predict(X_perm)
                perm_accuracy = sum(predictions == y_test)
            else:
                print("Model not defined")
                return 
            accs[trial,idx] = perm_accuracy 
            losses[trial,idx] = perm_loss
                     
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

    plt.barh(y_pos,accs_mean)
    plt.title("Accuracy")
    plt.yticks(y_pos,labels)
    plt.show()

    return accs