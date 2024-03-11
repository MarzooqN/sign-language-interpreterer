from keras.utils import plot_model
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
import numpy as np
from neural_network import create_model, create_model_MS
from collect_data import Data

def evaluate(Data: Data, X_test, y_test):
    model = create_model(Data)

    model.load_weights('actions.keras')


    plot_model(model, to_file="model_image.png", show_shapes=True, show_layer_names=True)


    yhat = model.predict(X_test)

    #Changes values from [0,1,0] etc. to 1, 2, or 3
    ytrue = np.argmax(y_test, axis=1).tolist()
    yhat = np.argmax(yhat, axis=1).tolist()

    #Gets a confusion matrix of size 2 2 in terms of [[TRUE N, FLASE P], [FLASE N, TRUE P]] so want values in true corners
    print(multilabel_confusion_matrix(ytrue, yhat))

    print(accuracy_score(ytrue, yhat))


def evaluate_MS(Data: Data, X_test, y_test):
    model = create_model_MS(Data)

    model.load_weights('actions.keras')


    plot_model(model, to_file="model_image.png", show_shapes=True, show_layer_names=True)


    yhat = model.predict(X_test)

    #Changes values from [0,1,0] etc. to 1, 2, or 3
    ytrue = np.argmax(y_test, axis=1).tolist()
    yhat = np.argmax(yhat, axis=1).tolist()

    #Gets a confusion matrix of size 2 2 in terms of [[TRUE N, FLASE P], [FLASE N, TRUE P]] so want values in true corners
    print(multilabel_confusion_matrix(ytrue, yhat))

    print(accuracy_score(ytrue, yhat))
