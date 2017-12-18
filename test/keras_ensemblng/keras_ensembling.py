from keras.models import Model, Input
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Activation, Average, Dropout
from keras.utils import to_categorical
from keras.losses import categorical_crossentropy
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import Adam
from keras.datasets import cifar10
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import os


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Keras
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
set_session(tf.Session(config=config))

# Tensorflow
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
# print(sess.run(y))

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train / 255.
x_test = x_test / 255.
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)


print('x_train shape: {} | y_train shape: {}\nx_test shape : {} | y_test shape : {}'.format(x_train.shape, y_train.shape,x_test.shape, y_test.shape))


input_shape = x_train[0,:,:,:].shape
model_input = Input(shape=input_shape)

# 一定要放在前面，呼叫才能辨識到
def compile_and_train(model, num_epochs): 
    model.compile(loss=categorical_crossentropy, optimizer=Adam(), metrics=['acc']) 
    filepath = 'weights/' + model.name + '.{epoch:02d}-{loss:.2f}.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=0, save_weights_only=True, save_best_only=True, mode='auto', period=1)
    tensor_board = TensorBoard(log_dir='logs/', histogram_freq=0, batch_size=32)
    history = model.fit(x=x_train, y=y_train, batch_size=32, epochs=num_epochs, verbose=1, callbacks=[checkpoint, tensor_board], validation_split=0.2)
    return history

def test(model):
    pred = model.predict(x_test, batch_size = 32)
    pred = np.argmax(pred, axis=1)
    error = np.sum(pred != y_test) / y_test.shape[0]
    return error

def evaluate_error(model):
    pred = model.predict(x_test, batch_size = 32)
    pred = np.argmax(pred, axis=1)
    pred = np.expand_dims(pred, axis=1) # make same shape as y_test
    error = np.sum(np.not_equal(pred, y_test)) / y_test.shape[0]  
    return error

def conv_pool_cnn(model_input):
    x = Conv2D(96, kernel_size=(3, 3), activation='relu', padding = 'same')(model_input)
    x = Conv2D(96, (3, 3), activation='relu', padding = 'same')(x)
    x = Conv2D(96, (3, 3), activation='relu', padding = 'same')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides = 2)(x)
    x = Conv2D(192, (3, 3), activation='relu', padding = 'same')(x)
    x = Conv2D(192, (3, 3), activation='relu', padding = 'same')(x)
    x = Conv2D(192, (3, 3), activation='relu', padding = 'same')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides = 2)(x)
    x = Conv2D(192, (3, 3), activation='relu', padding = 'same')(x)
    x = Conv2D(192, (1, 1), activation='relu')(x)
    x = Conv2D(10, (1, 1), activation='relu')(x)
    x = GlobalAveragePooling2D('channels_last')(x)
    x = Activation(activation='softmax')(x)
    model = Model(model_input, x, name='conv_pool_cnn')
    return model

conv_pool_cnn_model = conv_pool_cnn(model_input)
conv_pool_cnn_model.summary()

_ = compile_and_train(conv_pool_cnn_model, num_epochs=20)

'''Access the loss and accuracy in every epoch'''
loss = _.history.get('loss')
acc = _.history.get('acc')

''' Access the performance on validation data '''
val_loss = _.history.get('val_loss')
val_acc = _.history.get('val_acc')

''' Visualize the loss and accuracy of both models'''
plt.figure(1)
plt.subplot(121)
plt.plot(range(len(loss)), loss,label='Training')
plt.plot(range(len(val_loss)), val_loss,label='Validation')
plt.title('Loss')
plt.legend(loc='upper left')
plt.subplot(122)
plt.plot(range(len(acc)), acc,label='Training')
plt.plot(range(len(val_acc)), val_acc,label='Validation')
plt.title('Accuracy')
plt.show()

print(test(conv_pool_cnn_model))
print(evaluate_error(conv_pool_cnn_model))

def all_cnn(model_input):
    x = Conv2D(96, kernel_size=(3, 3), activation='relu', padding = 'same')(model_input)
    x = Conv2D(96, (3, 3), activation='relu', padding = 'same')(x)
    x = Conv2D(96, (3, 3), activation='relu', padding = 'same', strides = 2)(x)
    x = Conv2D(192, (3, 3), activation='relu', padding = 'same')(x)
    x = Conv2D(192, (3, 3), activation='relu', padding = 'same')(x)
    x = Conv2D(192, (3, 3), activation='relu', padding = 'same', strides = 2)(x)
    x = Conv2D(192, (3, 3), activation='relu', padding = 'same')(x)
    x = Conv2D(192, (1, 1), activation='relu')(x)
    x = Conv2D(10, (1, 1), activation='relu')(x)
    x = GlobalAveragePooling2D('channels_last')(x)
    x = Activation(activation='softmax')(x)
    model = Model(model_input, x, name='all_cnn')
    return model

all_cnn_model = all_cnn(model_input)
all_cnn_model.summary()

_ = compile_and_train(all_cnn_model, num_epochs=20)

'''Access the loss and accuracy in every epoch'''
loss = _.history.get('loss')
acc = _.history.get('acc')

''' Access the performance on validation data '''
val_loss = _.history.get('val_loss')
val_acc = _.history.get('val_acc')

''' Visualize the loss and accuracy of both models'''
plt.figure(2)
plt.subplot(121)
plt.plot(range(len(loss)), loss,label='Training')
plt.plot(range(len(val_loss)), val_loss,label='Validation')
plt.title('Loss')
plt.legend(loc='upper left')
plt.subplot(122)
plt.plot(range(len(acc)), acc,label='Training')
plt.plot(range(len(val_acc)), val_acc,label='Validation')
plt.title('Accuracy')
plt.show()

print(test(all_cnn_model))
print(evaluate_error(all_cnn_model))

def nin_cnn(model_input):
    #mlpconv block 1
    x = Conv2D(32, (5, 5), activation='relu',padding='valid')(model_input)
    x = Conv2D(32, (1, 1), activation='relu')(x)
    x = Conv2D(32, (1, 1), activation='relu')(x)
    x = MaxPooling2D((2,2))(x)
    x = Dropout(0.5)(x)
    #mlpconv block2
    x = Conv2D(64, (3, 3), activation='relu',padding='valid')(x)
    x = Conv2D(64, (1, 1), activation='relu')(x)
    x = Conv2D(64, (1, 1), activation='relu')(x)
    x = MaxPooling2D((2,2))(x)
    x = Dropout(0.5)(x)
    #mlpconv block3
    x = Conv2D(128, (3, 3), activation='relu',padding='valid')(x)
    x = Conv2D(32, (1, 1), activation='relu')(x)
    x = Conv2D(10, (1, 1), activation='relu')(x)
    x = GlobalAveragePooling2D()(x)
    x = Activation(activation='softmax')(x)
    model = Model(model_input, x, name='nin_cnn')
    return model

nin_cnn_model = nin_cnn(model_input)
nin_cnn_model.summary()


_ = compile_and_train(nin_cnn_model, num_epochs=20)

'''Access the loss and accuracy in every epoch'''
loss = _.history.get('loss')
acc = _.history.get('acc')

''' Access the performance on validation data '''
val_loss = _.history.get('val_loss')
val_acc = _.history.get('val_acc')

''' Visualize the loss and accuracy of both models'''
plt.figure(3)
plt.subplot(121)
plt.plot(range(len(loss)), loss,label='Training')
plt.plot(range(len(val_loss)), val_loss,label='Validation')
plt.title('Loss')
plt.legend(loc='upper left')
plt.subplot(122)
plt.plot(range(len(acc)), acc,label='Training')
plt.plot(range(len(val_acc)), val_acc,label='Validation')
plt.title('Accuracy')
plt.show()

print(test(nin_cnn_model))
print(evaluate_error(nin_cnn_model))

def ensemble(models, model_input):
    outputs = [model.outputs[0] for model in models]
    y = Average()(outputs)
    model = Model(model_input, y, name='ensemble')
    return model

models = [conv_pool_cnn_model, all_cnn_model, nin_cnn_model]
ensemble_model = ensemble(models, model_input)
ensemble_model.summary()

_ = compile_and_train(ensemble_model, num_epochs=20)

'''Access the loss and accuracy in every epoch'''
loss = _.history.get('loss')
acc = _.history.get('acc')

''' Access the performance on validation data '''
val_loss = _.history.get('val_loss')
val_acc = _.history.get('val_acc')

''' Visualize the loss and accuracy of both models'''
plt.figure(4)
plt.subplot(121)
plt.plot(range(len(loss)), loss,label='Training')
plt.plot(range(len(val_loss)), val_loss,label='Validation')
plt.title('Loss')
plt.legend(loc='upper left')
plt.subplot(122)
plt.plot(range(len(acc)), acc,label='Training')
plt.plot(range(len(val_acc)), val_acc,label='Validation')
plt.title('Accuracy')
plt.show()

print(test(ensemble_model))
print(evaluate_error(ensemble_model))