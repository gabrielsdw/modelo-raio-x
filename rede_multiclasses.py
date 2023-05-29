from keras import regularizers as rg
from keras import layers
from keras.preprocessing import image
import tensorflow as tf
import matplotlib.pyplot as plt

path_train = r"D:\teste_dataset_modificado2\train"
path_test = r"D:\teste_dataset_modificado2\test"

datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=10,     
    width_shift_range=0.1,  
    height_shift_range=0.1,   
    horizontal_flip=True,   
    fill_mode='nearest'     
)

augmented_generator = datagen.flow_from_directory(
    directory=path_train,
    shuffle=True,
    batch_size=32,
    class_mode="categorical",
    color_mode="rgb",
    target_size=(64, 64)
)

train_data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0/255    
)

test_data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0/255
)

train_generator = train_data_gen.flow_from_directory(
    directory=path_train,
    shuffle=True,
    batch_size=32,
    class_mode='categorical',
    color_mode="rgb",
    target_size=(64, 64),
)

test_generator = test_data_gen.flow_from_directory(
    directory=path_test,
    shuffle=True,
    batch_size=32,
    class_mode='categorical',
    color_mode="rgb",
    target_size=(64, 64),
)
print(train_generator.class_indices)
print(test_generator.class_indices)


model = tf.keras.models.Sequential([
    layers.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", input_shape=(64, 64, 3)),
    #layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu"),
    #layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu"),
    #layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    
    layers.Dense(units=128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(units=128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(units=128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    #layers.Dense(units=32, activation='relu'),
    #layers.BatchNormalization(),
    #layers.Dropout(0.5),
    
    #layers.Dense(units=32, activation='relu', kernel_regularizer=rg.l1(0.001)),
    #layers.Dropout(0.5),
    layers.Dense(units=5, activation='softmax'),
])

print(model.summary())

loss = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)

model.compile(
    loss=loss,
    optimizer=optimizer,
    metrics=['accuracy']
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5
)

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    'melhor_modelo_rede5class_01.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max'
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    patience=5,
    min_lr=0.0035,
    factor=0.02
)

history = model.fit(
    epochs=200,
    x=augmented_generator,
    validation_data=test_generator,
    shuffle=True,
    callbacks=[checkpoint_callback],
    verbose=1,
    batch_size=32
)

# plotar a perda de treinamento e validação ao longo do tempo
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Perda do Modelo')
plt.ylabel('Perda')
plt.xlabel('Época')
plt.legend(['Treinamento', 'Validação'], loc='upper right')
plt.show()

# plotar a precisão de treinamento e validação ao longo do tempo
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Precisão do Modelo')
plt.ylabel('Precisão')
plt.xlabel('Época')
plt.legend(['Treinamento', 'Validação'], loc='lower right')
plt.show()
