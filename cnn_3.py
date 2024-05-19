import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, BatchNormalization, ReLU, Dropout, Flatten, Dense, Add, Concatenate, AveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
import time

# Enable mixed precision
tf.keras.mixed_precision.set_global_policy('mixed_float16')

def create_time_series_classifier(input_shape, num_classes, dropout_rate=0.2):
    def skip_connection(conv_out, skip_tensor):
        x = MaxPooling1D(pool_size=2, strides=1)(skip_tensor)
        x = Concatenate()([conv_out, x])
        return x
    
    def conv_block(input_tensor, filters):
        x = BatchNormalization()(input_tensor)
        x = ReLU()(x)
        x = Dropout(dropout_rate)(x)
        x = Conv1D(filters=filters, kernel_size=15, padding='same', 
                    data_format="channels_last",
                    kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
        x = ReLU()(x)
        x = AveragePooling1D(pool_size=2, strides=1)(x)
        return x
    
    inputs = Input(shape=input_shape)
    # Try different kernel size 
 
    # Starting with 8 filters 
    filters = 64
    x = conv_block(inputs, filters)
    x = skip_connection(x, inputs)
    for i in range(15):
        skip_tensor = x
        if i <=2 and i >=0: # Increase the number of filters in each block
            filters = 64
            print('filters',filters,'iter',i)
        elif i <=6 and i >=3:
            filters = 32
            print('filters',filters,'iter',i)
        elif i <=10 and i >=7:
            filters = 16
            print('filters',filters,'iter',i)
        elif i <=14 and i >=11:
            filters = 8
            print('filters',filters,'iter',i)
        x = conv_block(x, filters)
        x = skip_connection(x, skip_tensor)
    
    x = ReLU()(x)
    x = Flatten()(x)
    # x = Dropout(0.5)(x)
    x = Dense(126,activation="sigmoid")(x)
    outputs = Dense(num_classes, activation='softmax', 
                    kernel_regularizer=tf.keras.regularizers.l2(0.01),
                    dtype='float32')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

data_dir = '/content/drive/MyDrive/Elena-Seminar/'
data_name = 'mydata1000'
output_dir = f'results_cnn_anapoda_filters/{data_name}/'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def load_data(data_dir, data_name):
    train_data = pd.read_csv(data_dir + data_name + f'/{data_name}_train.csv', header=None)
    val_data = pd.read_csv(data_dir + data_name + f'/{data_name}_val.csv', header=None)
    test_data = pd.read_csv(data_dir + data_name + f'/{data_name}_test.csv', header=None)

    train_labels = train_data.iloc[:, 0].values
    train_signals = train_data.iloc[:, 1:].values

    val_labels = val_data.iloc[:, 0].values
    val_signals = val_data.iloc[:, 1:].values

    test_labels = test_data.iloc[:, 0].values
    test_signals = test_data.iloc[:, 1:].values

    # Reshape in 3D array format
    train_signals = train_signals.reshape(-1, train_signals.shape[1], 1)
    val_signals = val_signals.reshape(-1, val_signals.shape[1], 1)
    test_signals = test_signals.reshape(-1, test_signals.shape[1], 1)
    
    return train_signals, train_labels, val_signals, val_labels, test_signals, test_labels

train_signals, train_labels, val_signals, val_labels, test_signals, test_labels = load_data(data_dir, data_name)

def calculate_metrics(y_true, y_pred, duration):
    res = pd.DataFrame(data=np.zeros((1, 16), dtype=float), index=[0],
                       columns=['precision', 'accuracy', 'recall', 
                                'precision_0', 'accuracy_0', 'recall_0',
                                'precision_1', 'accuracy_1', 'recall_1',
                                'precision_2', 'accuracy_2', 'recall_2',
                                'precision_3', 'accuracy_3', 'recall_3',
                                'duration'])
    
    res['precision'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    res['accuracy'] = accuracy_score(y_true, y_pred)
    res['recall'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    
    for i in range(4):
        idx = np.where(np.array(y_true) == i)[0]
        if len(idx) == 0:
            res[f'precision_{i}'] = 0
            res[f'accuracy_{i}'] = 0
            res[f'recall_{i}'] = 0
        else:
            res[f'precision_{i}'] = precision_score(np.array(y_true)[idx], np.array(y_pred)[idx], average='macro',zero_division=0)
            res[f'accuracy_{i}'] = accuracy_score(np.array(y_true)[idx], np.array(y_pred)[idx])
            res[f'recall_{i}'] = recall_score(np.array(y_true)[idx], np.array(y_pred)[idx], average='macro', zero_division=0)

    res['duration'] = duration
    return res


# ######
# ######
# #CROSS VALIDATION

# input_shape = train_signals.shape[1:]  
# num_classes = len(np.unique(train_labels))

# num_folds = 5  # Number of folds for cross-validation
# accuracies = []
# precisions = []
# recalls = []

# skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

# for fold, (train_index, val_index) in enumerate(skf.split(train_signals, train_labels)):
#     print(f'Fold {fold + 1}/{num_folds}')
    
#     # Split data into train and validation sets for this fold
#     x_train, x_val = train_signals[train_index], train_signals[val_index]
#     y_train, y_val = train_labels[train_index], train_labels[val_index]

#     # Further split the training data into 80% train and 20% validation
#     split_index = int(len(x_train) * 0.8)
#     x_train_split, x_val_split = x_train[:split_index], x_train[split_index:]
#     y_train_split, y_val_split = y_train[:split_index], y_train[split_index:]

#     # Create and compile the model
#     model = create_time_series_classifier(input_shape, num_classes)
#     optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
#     model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

   
#     early_stopping = EarlyStopping(monitor='val_loss', patience=50, verbose=1)
#     checkpoint_path = os.path.join(output_dir, f'best_model_fold{fold + 1}.h5')
#     model_checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True)

   
#     start_time = time.time()
#     hist = model.fit(x_train_split, y_train_split, epochs=1500, batch_size=4, 
#                      validation_data=(x_val_split, y_val_split), callbacks=[early_stopping, model_checkpoint])
#     duration = time.time() - start_time
#     print('Fit time:', duration)

#     # Load the best model
#     model.load_weights(checkpoint_path)

#     test_loss, test_accuracy = model.evaluate(test_signals, test_labels)
#     print(f"Fold {fold + 1} - Test loss: {test_loss:.4f}, Test accuracy: {test_accuracy:.4f}")


#     y_pred = model.predict(test_signals)
#     y_pred_classes = np.argmax(y_pred, axis=1)

   
#     hist_df = pd.DataFrame(hist.history)
#     hist_df.to_csv(output_dir + f'history_fold{fold + 1}.csv', index=False)
#     df_metrics = calculate_metrics(test_labels, y_pred_classes, duration)
#     df_metrics.to_csv(output_dir + f'df_metrics_fold{fold + 1}.csv', index=False)

#     accuracies.append(df_metrics['accuracy'].values[0])
#     precisions.append(df_metrics['precision'].values[0])
#     recalls.append(df_metrics['recall'].values[0])

# mean_accuracy = np.mean(accuracies)
# mean_precision = np.mean(precisions)
# mean_recall = np.mean(recalls)

# print("Mean Accuracy:", mean_accuracy)
# print("Mean Precision:", mean_precision)
# print("Mean Recall:", mean_recall)

# data = {
#     'Metric': ['Mean Accuracy', 'Mean Precision', 'Mean Recall'],
#     'Value': [mean_accuracy, mean_precision, mean_recall]
# }
# df = pd.DataFrame(data)

# overall_dir = f'results_cnn/{data_name}/overall'  

# if not os.path.exists(overall_dir):
#     os.makedirs(overall_dir)

# file_path = os.path.join(overall_dir, 'overall_metrics.csv')
# df.to_csv(file_path, index=False)



######
######
#NO CROSS VALIDATION
input_shape = train_signals.shape[1:]  
num_classes = len(np.unique(train_labels))
# from sklearn.utils.class_weight import compute_class_weight
# class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_labels), 
                                      # y=train_labels)
# class_weights_dict = dict(enumerate(class_weights))

num_iterations = 5
accuracies = []
precisions = []
recalls = []

for i in range(num_iterations):
    model = create_time_series_classifier(input_shape, num_classes)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=50, verbose=1)

    # checkpoint_path = os.path.join(output_dir, f'best_model_iter{i+1}.h5')
    checkpoint_path = os.path.join(output_dir, f'best_model.keras')
    model_checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True)

    start_time = time.time()
    
    hist = model.fit(train_signals, train_labels, epochs=100, batch_size=32, 
                     validation_data=(val_signals, val_labels),
                     class_weight=class_weights_dict,
                      callbacks=[early_stopping, model_checkpoint])
    duration = time.time() - start_time

    print('Fit time:', duration)

    model.load_weights(checkpoint_path)

    # Evaluate on test set
    test_loss, test_accuracy = model.evaluate(test_signals, test_labels)
    print(f"Iteration {i+1} - Test loss: {test_loss:.4f}, Test accuracy: {test_accuracy:.4f}")

    y_pred = model.predict(test_signals)
    y_pred_classes = np.argmax(y_pred, axis=1)

    hist_df = pd.DataFrame(hist.history)
    hist_df.to_csv(output_dir + f'history_iter{i+1}.csv', index=False)
    df_metrics = calculate_metrics(test_labels, y_pred_classes, duration)
    df_metrics.to_csv(output_dir + f'df_metrics_iter{i+1}.csv', index=False)

    accuracies.append(df_metrics['accuracy'].values[0])
    precisions.append(df_metrics['precision'].values[0])
    recalls.append(df_metrics['recall'].values[0])

# Calculate mean metrics
mean_accuracy = np.mean(accuracies)
mean_precision = np.mean(precisions)
mean_recall = np.mean(recalls)

print("Mean Accuracy:", mean_accuracy)
print("Mean Precision:", mean_precision)
print("Mean Recall:", mean_recall)

data = {
    'Metric': ['Mean Accuracy', 'Mean Precision', 'Mean Recall'],
    'Value': [mean_accuracy, mean_precision, mean_recall]
}
df = pd.DataFrame(data)

overall_dir = f'results_cnn/{data_name}/overall'  

if not os.path.exists(overall_dir):
    os.makedirs(overall_dir)

file_path = os.path.join(overall_dir, 'overall_metrics.csv')
df.to_csv(file_path, index=False)