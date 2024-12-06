from keras import optimizers, losses, callbacks, metrics, mixed_precision
mixed_precision.set_global_policy("mixed_float16")

from model import get_model
from dataset import get_data_train_test_split


# Retrieve data
mask = [1, 5, 6, 7, 8, 9, 10, 11]
x_train, y_train, _, _ = get_data_train_test_split()
x_train = x_train[:, :, mask]

# Model Configurations
model_id = 16

lr = 1e-3
opt = optimizers.Adam(lr)
loss = losses.MeanSquaredError()

callbacks = [
    # Learning Optimizers
    callbacks.EarlyStopping(patience=9, min_delta=0.00001),
    callbacks.ReduceLROnPlateau(patience=7, min_lr=lr / 100),
    # Checkpoints
    callbacks.ModelCheckpoint(f"models/backups/model_best_{model_id}.keras",
                              save_best_only=True),
]

# Compile the model
model = get_model(x_train.shape[2])
model.compile(optimizer=opt, loss=loss,
              metrics=[metrics.mean_squared_error, metrics.mean_absolute_error])

# Train the model
history = model.fit(
    x_train, y_train,
    epochs=200,
    batch_size=1,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1,
)

# Save final result
model.save(f"models/model_{model_id}.keras")
