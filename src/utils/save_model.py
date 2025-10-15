# Save model to disk in pickle format
import joblib

import constants.paths as pth


def save_model(model, model_name, output_dir=pth.MODELS_DIR):
    filepath = f"{output_dir}/{model_name}.pkl"
    joblib.dump(model, filepath)
