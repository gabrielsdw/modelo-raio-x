from utils import showGraphic
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Dropout, BatchNormalization, MaxPooling2D, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.losses import BinaryCrossentropy
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint
from keras.regularizers import L2

batch_size = 32
loss = BinaryCrossentropy()
optimizer = SGD()
epochs = 300

path_train = r"D:\dataset2classes - Copia\train_train"
path_test = r"D:\dataset2classes - Copia\test_test"

datagen_train = ImageDataGenerator(
    rescale=1.0/255,
    horizontal_flip=True,
    rotation_range=10,
    fill_mode="nearest",
    width_shift_range=0.1,
    height_shift_range=0.1
)

datagen_test = ImageDataGenerator(
    rescale=1.0/255
)

train_generator = datagen_train.flow_from_directory(
    directory=path_train,
    batch_size=batch_size,
    class_mode="binary",
    color_mode="rgb",
    shuffle=True,
    target_size=(64, 64)
)

test_generator = datagen_test.flow_from_directory(
    directory=path_test,
    batch_size=batch_size,
    class_mode="binary",
    color_mode="rgb",
    shuffle=True,
    target_size=(64, 64)
)

print(train_generator.class_indices)
print(test_generator.class_indices)

model = Sequential([
    Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same", input_shape=(64, 64, 3)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same"),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same"),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(units=32, activation="relu", kernel_regularizer=L2(0.0001), kernel_initializer='he_normal'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(units=1, activation="sigmoid")
])

save_model = ModelCheckpoint(
    filepath="checkpoints/teste7.h5",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max"
)

model.compile( 
    loss=loss,
    optimizer=optimizer,
    metrics=['accuracy']
)

history = model.fit(
    epochs=epochs,
    batch_size=batch_size,
    x=train_generator,
    validation_data=test_generator,
    shuffle=True,
    callbacks=[save_model]
)

showGraphic(history)
