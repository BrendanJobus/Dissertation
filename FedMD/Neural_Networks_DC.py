from tensorflow.keras.models import Model, Sequential, clone_model, load_model
from tensorflow.keras.layers import Input, Dense, add, concatenate, Conv2D,Dropout,\
BatchNormalization, Flatten, MaxPooling2D, AveragePooling2D, Activation, Dropout, Reshape
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.python.trackable.data_structures import ListWrapper
import keras
import numpy as np
from tensorflow.python.ops import math_ops
import tensorflow as tf
import errno, os
import warnings
import copy

def cnn_3layer_fc_model(n_classes,n1 = 128, n2=192, n3=256, dropout_rate = 0.2,input_shape = (28,28)):
    model_A, x = None, None
     
    x = Input(input_shape)
    if len(input_shape)==2: 
        y = Reshape((input_shape[0], input_shape[1], 1))(x)
    else:
        y = Reshape(input_shape)(x)
    y = Conv2D(filters = n1, kernel_size = (3,3), strides = 1, padding = "same", 
            activation = None)(y)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)
    y = Dropout(dropout_rate)(y)
    y = AveragePooling2D(pool_size = (2,2), strides = 1, padding = "same")(y)

    y = Conv2D(filters = n2, kernel_size = (2,2), strides = 2, padding = "valid", 
            activation = None)(y)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)
    y = Dropout(dropout_rate)(y)
    y = AveragePooling2D(pool_size = (2,2), strides = 2, padding = "valid")(y)

    y = Conv2D(filters = n3, kernel_size = (3,3), strides = 2, padding = "valid", 
            activation = None)(y)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)
    y = Dropout(dropout_rate)(y)
    #y = AveragePooling2D(pool_size = (2,2), strides = 2, padding = "valid")(y)

    y = Flatten()(y)
    y = Dense(units = n_classes, activation = None, use_bias = False,
            kernel_regularizer=tf.keras.regularizers.l2(1e-3))(y)
    y = Activation("softmax")(y)


    model_A = DriftCorrectionModel(inputs = x, outputs = y)

    model_A.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate = 1e-3), 
                        loss = "sparse_categorical_crossentropy",
                        metrics = ["accuracy"])
    return model_A
  
def cnn_2layer_fc_model(n_classes, n1 = 128, n2=256, dropout_rate = 0.2, input_shape = (28,28)):
    model_A, x = None, None
    
    x = Input(input_shape)
    if len(input_shape)==2: 
        y = Reshape((input_shape[0], input_shape[1], 1))(x)
    else:
        y = Reshape(input_shape)(x)
    y = Conv2D(filters = n1, kernel_size = (3,3), strides = 1, padding = "same", 
            activation = None)(y)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)
    y = Dropout(dropout_rate)(y)
    y = AveragePooling2D(pool_size = (2,2), strides = 1, padding = "same")(y)


    y = Conv2D(filters = n2, kernel_size = (3,3), strides = 2, padding = "valid", 
            activation = None)(y)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)
    y = Dropout(dropout_rate)(y)
    #y = AveragePooling2D(pool_size = (2,2), strides = 2, padding = "valid")(y)

    y = Flatten()(y)
    y = Dense(units = n_classes, activation = None, use_bias = False,
            kernel_regularizer=tf.keras.regularizers.l2(1e-3))(y)
    y = Activation("softmax")(y)

    model_A = DriftCorrectionModel(inputs = x, outputs = y)

    model_A.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate = 1e-3), 
                        loss = "sparse_categorical_crossentropy",
                        metrics = ["accuracy"])
    
    return model_A


def remove_last_layer(model, loss = "mean_absolute_error"):
    """
    Input: Keras model, a classification model whose last layer is a softmax activation
    Output: Keras model, the same model with the last softmax activation layer removed,
        while keeping the same parameters 
    """
    
    new_model = Model(inputs = model.inputs, outputs = model.layers[-2].output)
    new_model.set_weights(model.get_weights())
    new_model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate = 1e-3), 
                      loss = loss)
    
    return new_model

###########################################################################################################################################
###########################################################################################################################################
###########################################################################################################################################

class DriftCorrectionModel(keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.global_model = 0
        self.hist_i = 0
        self._fed_dc_alpha_coef = 0
        self._run_drift_correction = False
        self.loss_cp = lambda theta_i : \
                        self._fed_dc_alpha_coef/2 * ((theta_i - (self.global_model - self.hist_i)) * (theta_i - (self.global_model - self.hist_i)))
        self.local_param_check = None

    def train_step(self, data):
        x, y, sample_weight = keras.src.engine.data_adapter.unpack_x_y_sample_weight(data)
        # Run forward pass.
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compute_loss(x, y, y_pred, sample_weight)
        # If we're looking to do drift correction, we need to change our loss function to our new objective function
        if self._run_drift_correction:
            loss = loss + np.sum( [ tf.reduce_sum(tf_var).numpy() for tf_var in self.loss_cp(self.trainable_variables) ] )

        self._validate_target_and_loss(y, loss)
        # Run backwards pass
        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        return self.compute_metrics(x, y, y_pred, sample_weight)
    
    def initiate_drift_correction_variables(self, alpha):
        self._fed_dc_alpha_coef = alpha
        self.hist_i = np.array(self.setting_hist_i, dtype=object)
        self._run_drif_correction = True
            
    def set_drift_correction_variables(self, w):
        self.global_model = np.array(w, dtype=object)

    @property
    def setting_hist_i(self):
        return [tf.zeros(shape=v.shape, name=v.name) for v in self.trainable_variables]
        
    def update_drift_correction_variables(self):
        current_model_params = np.array(self.trainable_variables, dtype=object)
        delta_param_curr = np.subtract(current_model_params, self.global_model, dtype=object)
        self.hist_i = np.add(self.hist_i, delta_param_curr, dtype=object)

def clone_subclassed_model(model, optimizer):
    clone = None
    clone = DriftCorrectionModel(inputs = model.inputs, outputs = model.output)
    clone.set_weights(model.get_weights())
    clone.compile(optimizer=optimizer, 
                        loss = "sparse_categorical_crossentropy",
                        metrics = ["accuracy"])
    return clone

###########################################################################################################################################
###########################################################################################################################################
###########################################################################################################################################

def train_models(models, X_train, y_train, X_test, y_test, 
                 save_dir = "./", save_names = None,
                 early_stopping = True, min_delta = 0.001, patience = 3, 
                 batch_size = 128, epochs = 20, is_shuffle=True, verbose = 1
                ):
    '''
    Train an array of models on the same dataset. 
    We use early termination to speed up training. 
    '''
    
    resulting_val_acc = []
    record_result = []
    for n, model in enumerate(models):
        print("Training model ", n)
        if early_stopping:
            model.fit(X_train, y_train, 
                      validation_data = [X_test, y_test],
                      callbacks=[EarlyStopping(monitor='val_accuracy', min_delta=min_delta, patience=patience)],
                      batch_size = batch_size, epochs = epochs, shuffle=is_shuffle, verbose = verbose
                     )
        else:
            model.fit(X_train, y_train, 
                      validation_data = [X_test, y_test],
                      batch_size = batch_size, epochs = epochs, shuffle=is_shuffle, verbose = verbose
                     )
        
        resulting_val_acc.append(model.history.history["val_accuracy"][-1])
        record_result.append({"train_acc": model.history.history["accuracy"], 
                              "val_acc": model.history.history["val_accuracy"],
                              "train_loss": model.history.history["loss"], 
                              "val_loss": model.history.history["val_loss"]})
                
###########################################################################################################################################
###########################################################################################################################################
###########################################################################################################################################
        #save_dir = "./PastModels/"
###########################################################################################################################################
###########################################################################################################################################
###########################################################################################################################################

        if save_dir is not None:
            save_dir_path = os.path.abspath(save_dir)
            #make dir
            try:
                os.makedirs(save_dir_path)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise    

            if save_names is None:
                file_name = save_dir + "model_{0}".format(n) + ".h5"
            else:
                file_name = save_dir + save_names[n] + ".h5"
                print(file_name)
            model.save(file_name)
    
    print("pre-train accuracy: ")
    print(resulting_val_acc)
        
    return record_result