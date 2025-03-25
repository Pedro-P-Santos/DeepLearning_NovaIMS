""""
Callbacks 
"""
import os
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, TensorBoard

def default_callbacks(
        log_directory = r"/Users/pedrosantos/Documents 2/Deep Learning/Projeto.keras",
        model_path = r"/Users/pedrosantos/Documents 2/Deep Learning/Projeto.keras",
        monitor_metrics="val_accuracy",
        patience_es = 3,
        patience_rlr = 5,
        min_lr = 0.0001
):
        os.makedirs(log_directory, exist_ok=True)

        return [
        TensorBoard(log_dir=log_directory),
        ModelCheckpoint(
            filepath = r"/Users/pedrosantos/Documents 2/Deep Learning/Projeto.keras",
            monitor=monitor_metrics,
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        ),
        EarlyStopping(
            monitor=monitor_metrics,
            patience=patience_es
        ),
        ReduceLROnPlateau(
            monitor=monitor_metrics,
            factor = 0.3,
            patience=patience_rlr,
            verbose=1,
            min_lr = min_lr
        )
    ]
