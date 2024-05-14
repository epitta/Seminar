import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, ReLU, Dropout, Conv1D, AveragePooling1D, Flatten, Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
import os

def create_time_series_classifier(input_shape, dropout_rate=0.2):
    def skip_connection(conv_out, skip_tensor):
        x = tf.keras.layers.MaxPooling1D(pool_size=2, strides=1)(skip_tensor)
        x = tf.keras.layers.Concatenate()([conv_out, x])
        return x
    
    def conv_block(input_tensor):
        x = tf.keras.layers.BatchNormalization()(input_tensor)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
        x = tf.keras.layers.Conv1D(filters=64, kernel_size=15, padding='same', data_format="channels_last")(x)
        x = tf.keras.layers.AveragePooling1D(pool_size=2, strides=1)(x)
        return x
    
    inputs = tf.keras.Input(shape=input_shape)
    
    x = conv_block(inputs)
    x = skip_connection(x, inputs)
    for _ in range(15):
        skip_tensor = x
        x = conv_block(x)
        x = skip_connection(x, skip_tensor)
    x = ReLU()(x)
    
    x = Flatten()(x)
    
    outputs = Dense(4, activation='softmax')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())
    return model

def load_data(data_prefix):

    train_file = os.path.join(data_prefix, "mydata1000_train.csv")
    val_file = os.path.join(data_prefix, "mydata1000_val.csv")
    test_file = os.path.join(data_prefix, "mydata1000_test.csv")

    data_train = np.loadtxt(train_file, delimiter=',')
    data_val = np.loadtxt(val_file, delimiter=',')
    data_test = np.loadtxt(test_file, delimiter=',')

    X_train = data_train[:, 1:]
    y_train = data_train[:, 0]

    X_val = data_val[:, 1:]
    y_val = data_val[:, 0]

    X_test = data_test[:, 1:]
    y_test = data_test[:, 0]

    num_features = X_train.shape[1]
    X_train = X_train.reshape((-1, num_features, 1))

    X_val = X_val.reshape((-1, num_features, 1))

    X_test = X_test.reshape((-1, num_features, 1))

    num_classes = len(np.unique(y_train))
    y_train = np.eye(num_classes)[y_train.astype(int)]
    y_val = np.eye(num_classes)[y_val.astype(int)]
    y_test = np.eye(num_classes)[y_test.astype(int)]

    return X_train, X_val, X_test, y_train, y_val, y_test

def calculate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=None)
    recall = recall_score(y_true, y_pred, average=None)
    return accuracy, precision, recall

# Perform five iterations
num_iterations = 5
accuracies = []
precisions = []
recalls = []

X_train, X_val, X_test, y_train, y_val, y_test = load_data('DATA_SADL\mydata1000')

input_shape = (X_train.shape[1], X_train.shape[2])

for i in range(num_iterations):
    model = create_time_series_classifier(input_shape)

    early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)

    model.fit(X_train, y_train, epochs=1500, batch_size=4, validation_data=(X_val, y_val), callbacks=[early_stopping])

    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Iteration {i+1} - Test loss: {test_loss:.4f}, Test accuracy: {test_accuracy:.4f}")

    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    accuracy, precision, recall = calculate_metrics(y_true_classes, y_pred_classes)
    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)

# Calculate mean metrics
mean_accuracy = np.mean(accuracies)
mean_precision = np.mean(precisions, axis=0)
mean_recall = np.mean(recalls, axis=0)

print("Mean Accuracy:", mean_accuracy)
print("Mean Precision:", mean_precision)
print("Mean Recall:", mean_recall)
