from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Flatten, Dropout
from keras import optimizers
import tensorflow as tf
import tensorflow_addons as tfa

def train():
    # prevent tensorflow from allocating the totality of a GPU memory
    config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
    sess = tf.compat.v1.Session(config=config)


    datagen = ImageDataGenerator(rescale=1./255.)


    train_generator = datagen.flow_from_directory(
    directory = "./training_dataset/",
    class_mode = "categorical",
    batch_size = 32,
    target_size = (224, 224))
    
    validation_generator=datagen.flow_from_directory(
    directory = "./validation_dataset/",
    class_mode = "categorical",
    batch_size = 32,
    target_size = (224, 224))


    for data_batch, labels_batch in train_generator:
        print('data batch shape:', data_batch.shape)
        print('labels batch shape:', labels_batch.shape)
        break

    for data_batch, labels_batch in validation_generator:
        print('data batch shape:', data_batch.shape)
        print('labels batch shape:', labels_batch.shape)
        break

    resnet_model = tf.keras.applications.ResNet50(
        include_top = False,
        weights = None,
        input_shape=(224, 224, 3),
        pooling = 'avg',
        classes = 2,
    )

    model = Sequential()
    model.add(resnet_model)

    # resnet fc layer
    model.add(Flatten(name='flatten'))
    model.add(Dense(1000, activation='relu', name='fc2'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='sigmoid'))

    model.summary()

    # hyper parameters
    BATCH_SIZE = 64
    EPOCHS = 30
    learning_rate = 8e-5

    # choose one of loss functions
    loss_function = tfa.losses.SigmoidFocalCrossEntropy(alpha=0.4, gamma=1.0)
    # loss_function = tf.keras.losses.BinaryCrossentropy()

    optimizer = optimizers.Adam(learning_rate=learning_rate)

    # configure loss function and optimiser for training
    model.compile(loss=loss_function, optimizer=optimizer, metrics=['accuracy'])

    # logdir = "./kaggle/working"
    # tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

    history = model.fit_generator(generator=train_generator,
    validation_data=validation_generator,
    validation_steps=50,
    epochs=EPOCHS,
    verbose=1
    )

    model.save('model_focal_loss.h5')
    # model.save('model_binary_cross_entropy.h5')

if __name__ == "__main__":
    train()