import tensorflow as tf
from keras import layers

path_train = r"C:\Users\Gabriel\Documents\datasets\Knee images X-ray - Copia\train"
path_test = r"C:\Users\Gabriel\Documents\datasets\Knee images X-ray - Copia\test"

train_data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0/255
)
test_data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0/255
)

train_generator = train_data_gen.flow_from_directory(
    directory=path_train,
    shuffle=True,
    batch_size=8,
    class_mode='categorical',
    color_mode="rgb",
    target_size=(64, 64),
    seed=7
)

test_generator = test_data_gen.flow_from_directory(
    directory=path_test,
    shuffle=True,
    batch_size=8,
    class_mode='categorical',
    color_mode="rgb",
    target_size=(64, 64),
    seed=7
)
print(train_generator.class_indices)
print(test_generator.class_indices)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", input_shape=(64, 64, 3)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=512, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(units=256, activation='relu'),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(units=128, activation='relu'),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(units=3, activation='softmax'),
])

loss = tf.keras.losses.CategoricalCrossentropy()
optmizer = tf.keras.optimizers.Adagrad(learning_rate=0.001)
model.compile(
    loss=loss,
    optimizer=optmizer,
    metrics=['accuracy']
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5
)

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    'TESTE_MODELO_REDE_CLASSIFICAÇÃO_qtd_knee.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max'
)

model.fit(
    epochs=30,
    x=train_generator,
    validation_data=test_generator,
    shuffle=True,
    batch_size=16,
    callbacks=[checkpoint_callback, early_stop]
)
