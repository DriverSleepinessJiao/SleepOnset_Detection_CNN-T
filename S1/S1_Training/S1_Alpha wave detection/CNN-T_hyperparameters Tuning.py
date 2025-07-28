import numpy as np
import random
import scipy.io as sio
from keras.layers import ReLU
from keras.utils import np_utils
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Activation, Dense, Dropout, Flatten, Input, Conv1D, Conv2D,MaxPooling1D, BatchNormalization, GlobalAveragePooling1D, LayerNormalization,MultiHeadAttention, Concatenate
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.callbacks import Callback, ReduceLROnPlateau, ModelCheckpoint,EarlyStopping
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score,classification_report, roc_curve, auc, cohen_kappa_score, f1_score
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import class_weight
from tensorflow.keras.layers import Reshape
import matplotlib.pyplot as plt
from keras.layers import Activation
import pandas as pd

# 加载数据
TrainAlphafile1 = r"E:\Data\S1\Training\Alpha wave detection\Positive_1_0.1.mat"
TrainAlpha1 = sio.loadmat(TrainAlphafile1)['AlphaTrain']

TrainAlphafile0 = r"E:\Data\S1\Training\Alpha wave detection\Negative_1_0.1.mat"
TrainAlpha0 = sio.loadmat(TrainAlphafile0)['WakeTrain']


x_train = np.concatenate((TrainAlpha1, TrainAlpha0), axis=0)
y_train = np.concatenate((np.ones(TrainAlpha1.shape[0]), np.zeros(TrainAlpha0.shape[0])), axis=0)


x_train = np.expand_dims(x_train, axis=2)

input_shape = x_train.shape[1] * x_train.shape[2]
x_train = x_train.reshape(x_train.shape[0], input_shape).astype('float32')

# One-hot encoding
y_train = np_utils.to_categorical(y_train, 2)

# Define positional encoding function
def positional_encoding(sequence_length, d_model):
    pos = np.arange(sequence_length)[:, np.newaxis]  # Position indices
    i = np.arange(d_model)[np.newaxis, :]  # Feature dimension indices
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))  # Calculate angle rates
    pos_encoding = pos * angle_rates  # Calculate position encoding
    pos_encoding[:, 0::2] = np.sin(pos_encoding[:, 0::2])  # Even indices use sin
    pos_encoding[:, 1::2] = np.cos(pos_encoding[:, 1::2])  # Odd indices use cos
    return pos_encoding

# Transformer encoder component
def transformer_encoder(inputs, num_heads=2, ff_dim=32):
    sequence_length = inputs.shape[1]  # Get sequence length dynamically
    pos_encoding = positional_encoding(sequence_length, inputs.shape[-1])  # Calculate positional encoding
    inputs += pos_encoding  # Add positional encoding
    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=ff_dim)(inputs, inputs)
    attention_output = Dropout(0.1)(attention_output)
    out1 = LayerNormalization(epsilon=1e-6)(inputs + attention_output)  # Residual connection
    ff_output = Dense(ff_dim, activation="relu")(out1)  # Feed-forward network
    ff_output = Dropout(0.1)(ff_output)
    ff_output = Dense(inputs.shape[-1], activation="relu")(ff_output)  # Output dimension same as input
    return LayerNormalization(epsilon=1e-6)(out1 + ff_output)  # Final output with residual connection



class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(np.argmax(y_train, axis=1)),
    y=np.argmax(y_train, axis=1)
)
class_weights = {0: class_weights[0], 1: class_weights[1]}


conv_filters_1 = [50, 100, 150]
kernel_sizes_1 = [100,200,300]
conv_filters_2 = [50, 100, 150]
kernel_sizes_2 = [100,200,300]


def create_model(conv_filter1, kernel_size1, conv_filter2, kernel_size2):
    inputs = Input(shape=(x_train.shape[1], 1))

    # Alpha 信号处理
    model_m_alpha = Conv1D(filters=conv_filter1, kernel_size=kernel_size1, padding='valid')(inputs)
    model_m_alpha = BatchNormalization()(model_m_alpha)
    model_m_alpha = Activation('relu')(model_m_alpha)
    model_m_alpha = Conv1D(filters=conv_filter2, kernel_size=kernel_size2, padding='same')(model_m_alpha)
    model_m_alpha = BatchNormalization()(model_m_alpha)
    model_m_alpha = Activation('relu')(model_m_alpha)
    model_m_alpha = MaxPooling1D(pool_size=2)(model_m_alpha)


    for _ in range(2):
        model_m_alpha = transformer_encoder(model_m_alpha)
    model_m_alpha = GlobalAveragePooling1D()(model_m_alpha)
    model_m_alpha = Dropout(0.5)(model_m_alpha)


    dense = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(model_m_alpha)

    dropout = Dropout(0.5)(dense)
    outputs = Dense(2, activation='softmax')(dropout)


    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=5e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    return model



best_accuracy = 0
best_f1_score = 0
best_params = {}

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


results_df = pd.DataFrame(columns=['Conv Filter 1', 'Kernel Size 1', 'Conv Filter 2', 'Kernel Size 2', 'Accuracy', 'F1 Score', 'Accuracy Std'])

for conv_filter1 in conv_filters_1:
    for kernel_size1 in kernel_sizes_1:
        for conv_filter2 in conv_filters_2:
            for kernel_size2 in kernel_sizes_2:
                print(f"Evaluating model with Conv Filters: {conv_filter1}, Kernel Size: {kernel_size1}, Conv Filters: {conv_filter2}, Kernel Size: {kernel_size2}")
                fold_accuracies = []
                fold_f1_scores = []
                fold_confusion_matrices = []

                for fold, (train_idx, val_idx) in enumerate(kfold.split(x_train, y_train.argmax(axis=1))):
                    x_train_fold, x_val_fold = x_train[train_idx], x_train[val_idx]
                    y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

                    model = create_model(conv_filter1, kernel_size1, conv_filter2, kernel_size2)
                    model_save_path = f'best_model_fold{fold + 1}.h5'
                    checkpoint = ModelCheckpoint(model_save_path, monitor='val_accuracy', save_best_only=True, verbose=1)
                    early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, verbose=1)
                    reduceLROnPlateau = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6,
                                                          verbose=1)
                    model.fit(x_train_fold, y_train_fold, batch_size=128, epochs=100, validation_data=(x_val_fold, y_val_fold),
                              callbacks=[checkpoint, early_stopping, reduceLROnPlateau], verbose=0, class_weight=class_weights)

                    # 评估验证集
                    val_pred = model.predict(x_val_fold)
                    y_val_pred = (val_pred[:, 1] > 0.5).astype(int)
                    acc = accuracy_score(y_val_fold.argmax(axis=1), y_val_pred)
                    f1 = f1_score(y_val_fold.argmax(axis=1), y_val_pred)
                    fold_accuracies.append(acc)
                    fold_f1_scores.append(f1)
                    conf_mat = confusion_matrix(y_val_fold.argmax(axis=1), y_val_pred)
                    fold_confusion_matrices.append(conf_mat)
                    print(f"Fold {fold+1} Confusion Matrix:\n{conf_mat}")
                    print(f"Fold {fold+1} Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")

                # 计算当前参数组合的平均验证准确率和F1分数
                mean_accuracy = np.mean(fold_accuracies)
                mean_f1_score = np.mean(fold_f1_scores)
                accuracy_std = np.std(fold_accuracies)
                print(f"Mean Validation Accuracy for Conv Filters: {conv_filter1}, Kernel Size: {kernel_size1}, Conv Filters: {conv_filter2}, Kernel Size: {kernel_size2} - {mean_accuracy:.4f}")
                print(f"Mean Validation F1 Score for Conv Filters: {conv_filter1}, Kernel Size: {kernel_size1}, Conv Filters: {conv_filter2}, Kernel Size: {kernel_size2} - {mean_f1_score:.4f}")
                print(f"Accuracy Std for Conv Filters: {conv_filter1}, Kernel Size: {kernel_size1}, Conv Filters: {conv_filter2}, Kernel Size: {kernel_size2} - {accuracy_std:.4f}")
                # 将结果添加到DataFrame中
                results_df = pd.concat([results_df, pd.DataFrame({
                    'Conv Filter 1': [conv_filter1],
                    'Kernel Size 1': [kernel_size1],
                    'Conv Filter 2': [conv_filter2],
                    'Kernel Size 2': [kernel_size2],
                    'Accuracy Mean': [mean_accuracy],
                    'Accuracy Std': [accuracy_std],
                    'F1 Score Mean': [mean_f1_score],

                })], ignore_index=True)

                # 更新最佳参数组合
                if mean_accuracy > best_accuracy or (mean_accuracy == best_accuracy and mean_f1_score > best_f1_score):
                    best_accuracy = max(mean_accuracy, best_accuracy)
                    best_f1_score = max(mean_f1_score, best_f1_score)
                    best_params = {'conv_filter1': conv_filter1, 'kernel_size1': kernel_size1,
                                   'conv_filter2': conv_filter2, 'kernel_size2': kernel_size2}



results_df.to_excel('S1CNNtrans-alpha -parameter_optimization_results.xlsx', index=False)

print(f"Best Params: {best_params} with Validation Accuracy: {best_accuracy:.4f} and F1 Score: {best_f1_score:.4f}")




