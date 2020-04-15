import matplotlib.pyplot as plt
import sklearn.datasets as skdata
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix

numeros = skdata.load_digits()
target = numeros['target']
imagenes = numeros['images']
n_imagenes = len(target)

data = imagenes.reshape((n_imagenes, -1)) # para volver a tener los datos como imagen basta hacer data.reshape((n_imagenes, 8, 8))
scaler = StandardScaler()
x_train, x_test, y_train, y_test = train_test_split(data, target, train_size=0.5)
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
clf = LogisticRegression(C=0.1, penalty='l1', solver='saga', tol=0.1)
clf.fit(x_train, y_train)

coef = clf.coef_.copy()
plt.figure(figsize=(10, 5))
scale = np.abs(coef).max()
for i in range(10):
    l1_plot = plt.subplot(2, 5, i + 1)
    l1_plot.imshow(coef[i].reshape(8, 8), interpolation='nearest',
                   cmap=plt.cm.RdBu, vmin=-scale, vmax=scale)
    l1_plot.set_xticks(())
    l1_plot.set_yticks(())
    l1_plot.set_xlabel('Número %i' % i)
plt.savefig('coeficientes.png')
plt.close()

plot_confusion_matrix(clf,x_test,y_test,normalize='true',values_format='.1f')
plt.title('Matriz de Confusión para C=0.1')
plt.show()
plt.savefig('confusion.png')