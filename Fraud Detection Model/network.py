import numpy as np
import tensorflow as tf
from keras import Sequential, layers, losses, optimizers, models, callbacks, regularizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler
import os
import pandas as pd

np.random.seed(64)
tf.random.set_seed(64)

class Predict:
    """
    A class to train and evaluate a fraud detection model using TensorFlow/Keras.

    Attributes:
        X_train (numpy.ndarray): Training feature set.
        Y_train (numpy.ndarray): Training target set.
        X_cv (numpy.ndarray): Cross-validation feature set.
        Y_cv (numpy.ndarray): Cross-validation target set.
        X_test (numpy.ndarray): Test feature set.
        Y_test (numpy.ndarray): Test target set.
        best_model (keras.Model): The best-performing model.
        best_model_history (dict): Information about the best model's performance.
        history (list): History of all models evaluated.
        accuracy (float): Accuracy of the best model on the test set.
    """

    def __init__(self, X, Y, start_learning_rate, feature_size, epochs, num_layers):
        """
        Initialize the Predict class, train the model, and save the best model.

        Args:
            X (numpy.ndarray): Input feature data.
            Y (numpy.ndarray): Target data.
            start_learning_rate (float): Initial learning rate for the optimizer.
            feature_size (int): Number of features in the dataset.
            epochs (int): Number of training epochs.
            num_layers (int): Maximum number of layers to evaluate.
        """
        self.X_train, self.Y_train, self.X_cv, self.Y_cv, self.X_test, self.Y_test = self.split_data(X, Y)
        packed = (start_learning_rate, feature_size, epochs, self.X_train, self.Y_train)
        self.best_model, self.best_model_history, self.history = self.select_optimal_model(
            packed, self.model_construction, self.X_cv, self.Y_cv, self.evaluate_model, max_layers=num_layers
        )
        self.accuracy = self.test_best_model(self.best_model, self.X_test, self.Y_test)
        self.get_best_model = (self.best_model, self.best_model_history, self.history, self.accuracy)
        self.best_model.save('fraud_detection_model.h5')

    def split_data(self, X, Y):
        """
        Split the data into training, cross-validation, and test sets.

        Args:
            X (numpy.ndarray): Input features.
            Y (numpy.ndarray): Target data.

        Returns:
            tuple: Split data (X_train, Y_train, X_cv, Y_cv, X_test, Y_test).
        """
        X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.4, random_state=42)
        X_cv, X_test, Y_cv, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)
        del X_temp, Y_temp
        return X_train, Y_train, X_cv, Y_cv, X_test, Y_test

    def model_construction(self, X_train, Y_train, X_cv, Y_cv, start_learning_rate, feature_size, epochs, num_layers, vlambda=0.01):
        """
        Construct and train a neural network model.

        Args:
            X_train (numpy.ndarray): Training features.
            Y_train (numpy.ndarray): Training labels.
            X_cv (numpy.ndarray): Validation features.
            Y_cv (numpy.ndarray): Validation labels.
            start_learning_rate (float): Initial learning rate for the optimizer.
            feature_size (int): Number of input features.
            epochs (int): Number of training epochs.
            num_layers (int): Number of hidden layers.
            vlambda (float): Regularization parameter (default: 0.01).

        Returns:
            tuple: The trained model and list of neurons in each layer.
        """
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_cv = scaler.transform(X_cv)

        layer_size = [np.random.choice(np.arange(8, 128, 8)) for _ in range(num_layers)]
        layer_list = [tf.keras.layers.InputLayer(input_shape=(feature_size,))]
        layer_list.append(layers.Dense(units=25, activation='relu', kernel_regularizer=regularizers.l2(vlambda)))
        layer_list.append(layers.Dropout(0.2))
        layer_list.append(layers.Dense(units=25, activation='relu', kernel_regularizer=regularizers.l2(vlambda)))
        layer_list.append(layers.Dropout(0.2))

        for i in range(num_layers):
            layer_list.append(layers.Dense(units=layer_size[i], activation='relu', kernel_regularizer=regularizers.l2(vlambda)))
            if i < num_layers - 1:
                layer_list.append(layers.Dropout(0.2))

        layer_list.append(layers.Dense(units=1, activation='sigmoid'))

        model = Sequential(layer_list)
        model.compile(
            optimizer=optimizers.Adam(learning_rate=start_learning_rate),
            loss=losses.BinaryCrossentropy(from_logits=False),
            metrics=['accuracy']
        )

        early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
        reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=5, min_lr=1e-6, verbose=1)
        model.fit(X_train, Y_train, epochs=epochs, validation_data=(X_cv, Y_cv), callbacks=[early_stopping, reduce_lr], verbose=1)

        return model, layer_size

    def evaluate_model(self, model, X_cv, Y_cv):
        """
        Evaluate a trained model on the cross-validation set.

        Args:
            model (keras.Model): The model to evaluate.
            X_cv (numpy.ndarray): Validation features.
            Y_cv (numpy.ndarray): Validation labels.

        Returns:
            tuple: Loss, accuracy, and F1-score of the model.
        """
        loss, accuracy = model.evaluate(X_cv, Y_cv, verbose=0)
        current_predictions = model.predict(X_cv)
        predicted_classes = (current_predictions > 0.5).astype(int)
        f1 = f1_score(Y_cv, predicted_classes, average='weighted')
        return loss, accuracy, f1

    def select_optimal_model(self, packed, model_construction, X_cv, Y_cv, evaluate_model, max_layers):
        """
        Train and select the best model based on performance metrics.

        Args:
            packed (tuple): Packed arguments (learning rate, feature size, epochs, X_train, Y_train).
            model_construction (function): Function to construct a model.
            X_cv (numpy.ndarray): Validation features.
            Y_cv (numpy.ndarray): Validation labels.
            evaluate_model (function): Function to evaluate a model.
            max_layers (int): Maximum number of layers to evaluate.

        Returns:
            tuple: Best model, its history, and the evaluation history of all models.
        """
        start_learning_rate, feature_size, epochs, X_train, Y_train = packed
        history = []
        best_model = None
        best_model_history = {}
        best_f1 = -float('inf')

        for i in range(1, max_layers + 1):
            current_model, layer_size = model_construction(X_train, Y_train, X_cv, Y_cv, start_learning_rate, feature_size, epochs, i)
            loss, accuracy, f1 = evaluate_model(current_model, X_cv, Y_cv)
            history.append({'Model': i, 'Loss': loss, 'Accuracy': accuracy, 'F1score': f1, 'Hidden Layers': i + 2, 'Neurons': layer_size})

            if f1 > best_f1:
                best_f1 = f1
                best_model = current_model
                best_model_history = {'Model': i, 'Loss': loss, 'Accuracy': accuracy, 'F1score': f1, 'Hidden Layers': i + 2, 'Neurons': layer_size}

        return best_model, best_model_history, history

    def test_best_model(self, best_model, X_test, Y_test):
        """
        Test the best model on the test set.

        Args:
            best_model (keras.Model): The best-performing model.
            X_test (numpy.ndarray): Test features.
            Y_test (numpy.ndarray): Test labels.

        Returns:
            float: Accuracy of the best model on the test set.
        """
        best_model_pred = best_model.predict(X_test)
        predicted_classes = (best_model_pred > 0.5).astype(int)
        accuracy = accuracy_score(Y_test, predicted_classes)
        print(f"Accuracy: {accuracy:.4f}")
        return accuracy

class RetrainModel:
    """
    Placeholder for retraining model functionality.
    """
    ...

def main():
    """
    Main function to load data, train the model, and evaluate it.
    """
    filename = "frauddetectiondataset.csv"
    if os.path.exists(filename):
        data = pd.read_csv(filename)
        data_encoded = pd.get_dummies(data, columns=['type'], drop_first=True)
        data_encoded = data_encoded.drop(columns=['nameOrig', 'nameDest'])
        X = data_encoded.drop(columns=['isFraud', 'isFlaggedFraud'])
        Y = data_encoded['isFraud']

        model_path = 'fraud_detection_model.h5'
        if os.path.exists(model_path):
            model = models.load_model(model_path)
            loss, accuracy = model.evaluate(X, Y, verbose=0)
            current_predictions = model.predict(X)
            predicted_classes = (current_predictions > 0.5).astype(int)
            f1 = f1_score(Y, predicted_classes, average='weighted')
            print(f"F1score: {f1}, Loss: {loss}, Accuracy: {accuracy}")
        else:
            start_learning_rate = 0.001
            feature_size = X.shape[1]
            epochs = 10
            num_layers = 2
            model = Predict(X, Y, start_learning_rate, feature_size, epochs, num_layers)
            best_model, best_model_history, history, accuracy = model.get_best_model
            print(f"Best Model History: {best_model_history} & Accuracy: {accuracy * 100:.1f}%")
            print(f"Total History: {history}")
    else:
        print("Missing Data")

if __name__ == "__main__":
    main()
