from __future__ import print_function
import numpy as np
import scipy.io as sio
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, classification_report
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape, GlobalAveragePooling1D
from keras.layers import Conv1D, MaxPooling1D, LSTM, Activation, BatchNormalization
from keras_self_attention import SeqSelfAttention
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.utils import np_utils
from keras import backend as K
from sklearn.utils import class_weight


TrainAlphafile1 = r"E:\Data\S1\Training\Alpha wave detection\Positive_1_0.1.mat"
TrainAlpha1 = sio.loadmat(TrainAlphafile1)['AlphaTrain']

TrainAlphafile0 = r"E:\Data\S1\Training\Alpha wave detection\Negative_1_0.1.mat"
TrainAlpha0 = sio.loadmat(TrainAlphafile0)['WakeTrain']


x_train = np.concatenate((TrainAlpha1, TrainAlpha0), axis=0)
y_train = np.concatenate((np.ones(TrainAlpha1.shape[0]), np.zeros(TrainAlpha0.shape[0])), axis=0)


x_train = np.expand_dims(x_train, axis=2)
input_shape = x_train.shape[1] * x_train.shape[2]
x_train = x_train.reshape(x_train.shape[0], input_shape).astype('float32')


y_train = np_utils.to_categorical(y_train, 2)


class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(np.argmax(y_train, axis=1)),
    y=np.argmax(y_train, axis=1)
)
class_weights = {0: class_weights[0], 1: class_weights[1]}


conv_filters_1 = [50, 150, 250]
kernel_sizes_1 = [50, 150, 250]
conv_filters_2 = [50, 150, 250]
kernel_sizes_2 = [50, 150, 250]
lstm_units = [50, 150, 250]



def create_model(conv_filter1, kernel_size1, conv_filter2, kernel_size2, lstm_unit):
    model = Sequential()
    model.add(Reshape((x_train.shape[1], 1), input_shape=(x_train.shape[1],)))
    model.add(Conv1D(conv_filter1, kernel_size1, strides=1, padding='valid'))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(MaxPooling1D(pool_size=3, strides=1))

    model.add(Conv1D(conv_filter2, kernel_size2, strides=1, padding='valid'))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(MaxPooling1D(pool_size=3, strides=1))

    model.add(LSTM(lstm_unit, return_sequences=True))
    model.add(SeqSelfAttention())
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.2))
    model.add(Dense(2, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model



best_accuracy = 0
best_f1_score = 0
best_params = {}

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


results_df = pd.DataFrame(
    columns=['Conv Filter 1', 'Kernel Size 1', 'Conv Filter 2', 'Kernel Size 2', 'LSTM Units', 'Accuracy', 'F1 Score'])

for conv_filter1 in conv_filters_1:
    for kernel_size1 in kernel_sizes_1:
        for conv_filter2 in conv_filters_2:
            for kernel_size2 in kernel_sizes_2:
                for lstm_unit in lstm_units:
                    print(
                        f"Evaluating model with Conv Filters: {conv_filter1}, Kernel Size: {kernel_size1}, Conv Filters: {conv_filter2}, Kernel Size: {kernel_size2}, LSTM Units: {lstm_unit}")
                    fold_accuracies = []
                    fold_f1_scores = []

                    for fold, (train_idx, val_idx) in enumerate(kfold.split(x_train, y_train.argmax(axis=1))):
                        x_train_fold, x_val_fold = x_train[train_idx], x_train[val_idx]
                        y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

                        model = create_model(conv_filter1, kernel_size1, conv_filter2, kernel_size2, lstm_unit)
                        model_save_path = f'best_model_fold{fold + 1}.h5'


                        checkpoint = ModelCheckpoint(model_save_path, monitor='val_accuracy', save_best_only=True,
                                                     verbose=1)
                        early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, verbose=1)
                        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6,
                                                      verbose=1)

                        callbacks = [checkpoint, early_stopping, reduce_lr]

                        model.fit(x_train_fold, y_train_fold,
                                  batch_size=150,
                                  epochs=50,
                                  validation_data=(x_val_fold, y_val_fold),
                                  callbacks=callbacks,
                                  verbose=0,
                                  class_weight=class_weights)


                        val_pred = model.predict(x_val_fold)
                        y_val_pred = (val_pred[:, 1] > 0.5).astype(int)
                        acc = accuracy_score(y_val_fold.argmax(axis=1), y_val_pred)
                        f1 = f1_score(y_val_fold.argmax(axis=1), y_val_pred)
                        fold_accuracies.append(acc)
                        fold_f1_scores.append(f1)


                    mean_accuracy = np.mean(fold_accuracies)
                    mean_f1_score = np.mean(fold_f1_scores)
                    print(
                        f"Mean Validation Accuracy for Conv Filters: {conv_filter1}, Kernel Size: {kernel_size1}, Conv Filters: {conv_filter2}, Kernel Size: {kernel_size2}, LSTM Units: {lstm_unit} - {mean_accuracy:.4f}")
                    print(
                        f"Mean Validation F1 Score for Conv Filters: {conv_filter1}, Kernel Size: {kernel_size1}, Conv Filters: {conv_filter2}, Kernel Size: {kernel_size2}, LSTM Units: {lstm_unit} - {mean_f1_score:.4f}")


                    results_df = pd.concat([results_df, pd.DataFrame({
                        'Conv Filter 1': [conv_filter1],
                        'Kernel Size 1': [kernel_size1],
                        'Conv Filter 2': [conv_filter2],
                        'Kernel Size 2': [kernel_size2],
                        'LSTM Units': [lstm_unit],
                        'Accuracy': [mean_accuracy],
                        'F1 Score': [mean_f1_score]
                    })], ignore_index=True)


                    if mean_accuracy > best_accuracy or (
                            mean_accuracy == best_accuracy and mean_f1_score > best_f1_score):
                        best_accuracy = max(mean_accuracy, best_accuracy)
                        best_f1_score = max(mean_f1_score, best_f1_score)
                        best_params = {'conv_filter1': conv_filter1, 'kernel_size1': kernel_size1,
                                       'conv_filter2': conv_filter2, 'kernel_size2': kernel_size2,
                                       'lstm_unit': lstm_unit}



print(f"Best Params: {best_params} with Validation Accuracy: {best_accuracy:.4f} and F1 Score: {best_f1_score:.4f}")