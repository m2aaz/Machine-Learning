# Fraud Detection Model

This repository contains code to train and evaluate a fraud detection model using deep learning techniques. The model is built with TensorFlow/Keras and utilizes a synthetic financial transaction dataset (e.g., Paysim1) to detect fraudulent transactions.

## Dataset

The model uses a dataset (e.g., Paysim1) that simulates financial transactions. Each record contains details about the transaction and the target variable isFraud, indicating whether a transaction is fraudulent (1) or not (0).

### Dataset Columns

* **nameOrig**: Originating account (removed for model training)
* **nameDest**: Destination account (removed for model training)
* **isFraud**: Target variable (1 for fraud, 0 for non-fraud)
* **isFlaggedFraud**: Flag for suspicious transactions
* **type**: The type of transaction (encoded as dummy variables)
* Additional features such as the transaction amount, customer details, and others

## Requirements

Ensure you have the following libraries installed:

* TensorFlow (for deep learning)
* Keras (part of TensorFlow)
* NumPy (for numerical operations)
* Pandas (for data manipulation)
* Scikit-learn (for model evaluation and preprocessing)

You can install the required dependencies using:

pip install tensorflow numpy pandas scikit-learn

## Dataset Preparation

### Download the Dataset

The dataset used in the code can be downloaded from [[link to dataset](https://www.kaggle.com/datasets/ealaxi/paysim1/data)]. The file should be named frauddetectiondataset.csv.

### Preprocessing

The dataset will be read using Pandas (pd.read_csv("frauddetectiondataset.csv")).
Categorical features such as type are one-hot encoded using pd.get_dummies().
Non-relevant columns like nameOrig and nameDest are dropped.
The target variable isFraud is separated from the features.

Example:

data_encoded = pd.get_dummies(data, columns=['type'], drop_first=True)
data_encoded = data_encoded.drop(columns=['nameOrig', 'nameDest'])
X = data_encoded.drop(columns=['isFraud', 'isFlaggedFraud'])
Y = data_encoded['isFraud']

## Model Overview 

### Neural Network Architecture

The model is a feed-forward neural network with:
* Multiple hidden layers (with varying sizes)
* Dropout layers to reduce overfitting
* Regularization to penalize large weights

### Training

* The model is compiled using the Adam optimizer with binary cross-entropy loss for binary classification
* Early stopping is implemented to prevent overfitting by monitoring the validation loss
* A learning rate scheduler adjusts the learning rate during training to optimize performance
* Keep in mind, restrict the models number of layers if you do not want to spend too long training and selecting the best model

### Model Evaluation

* The model is evaluated using accuracy, loss, and F1 score on a cross-validation set
* After training, the model is tested on a separate test set
* Model may use principals of neural architecture research to find the most optimal model

## Usage

### 1. Train the Model

If you have the dataset (frauddetectiondataset.csv), you can simply run the main() function. If the model has not been trained previously, it will be trained and saved as fraud_detection_model.h5. The code will print the evaluation metrics (Accuracy, F1 score, etc.) for both the best model and the final test set.

python fraud_detection.py

### 2. Use Pretrained Model

If the model is already saved as fraud_detection_model.h5, you can reload the model and evaluate it on new data using the following:

model = models.load_model('fraud_detection_model.h5')
loss, accuracy = model.evaluate(X, Y, verbose=0)
current_predictions = model.predict(X)
predicted_classes = (current_predictions > 0.5).astype(int)
f1 = f1_score(Y, predicted_classes, average='weighted')

### 3. Hyperparameter Tuning

The code includes functionality to experiment with different model architectures by varying the number of hidden layers and neurons in each layer. This is controlled by the num_layers parameter.

You can adjust the following hyperparameters:
* start_learning_rate (default: 0.001)
* epochs (default: 10)
* num_layers (default: 2)

### 4. Model File

The trained model will be saved as fraud_detection_model.h5. You can load it later to make predictions on new data or continue training.

## Key Functions

* **split_data**: Splits the dataset into training, validation, and test sets
* **model_construction**: Constructs and trains the deep learning model
* **evaluate_model**: Evaluates the model on the cross-validation set
* **select_optimal_model**: Trains multiple models with different configurations and selects the best-performing model
* **test_best_model**: Tests the best model on the test set and returns its accuracy

## License

Not Available.
