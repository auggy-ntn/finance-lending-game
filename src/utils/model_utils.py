# Save model to disk with appropriate format based on model type
import joblib
import lightgbm as lgb
import xgboost as xgb
from loguru import logger

import constants.paths as pth


def save_model(model, model_name, output_dir=pth.MODELS_DIR):
    """
    Save model using the appropriate serialization method.

    - XGBoost models: saved as .json (native format, best portability)
    - LightGBM models: saved as .txt (native format)
    - Scikit-learn models: saved as .pkl (joblib)

    Args:
        model: The trained model to save
        model_name: Name for the saved model file (without extension)
        output_dir: Directory to save the model (default: MODELS_DIR)

    Returns:
        filepath: Path where the model was saved
    """
    model_type = type(model).__module__

    if "xgboost" in model_type:
        filepath = f"{output_dir}/{model_name}.json"
        model.save_model(filepath)
    elif "lightgbm" in model_type:
        filepath = f"{output_dir}/{model_name}.txt"
        model.booster_.save_model(filepath)
    else:
        # Default to joblib for sklearn and other models
        filepath = f"{output_dir}/{model_name}.pkl"
        joblib.dump(model, filepath)

    logger.info(f"Model saved to: {filepath}")


def load_model(model_name, model_type, input_dir=pth.MODELS_DIR):
    """
    Load model from disk using the appropriate deserialization method.

    Args:
        model_name: Name of the saved model file (without extension)
        model_type: Type of the model ('xgboost', 'lightgbm', or 'sklearn')
        input_dir: Directory where the model is saved (default: MODELS_DIR)

    Returns:
        model: The loaded model
    """
    if model_type == "xgboost":
        filepath = f"{input_dir}/{model_name}.json"
        model = xgb.XGBClassifier()
        model.load_model(filepath)
    elif model_type == "lightgbm":
        filepath = f"{input_dir}/{model_name}.txt"
        model = lgb.LGBMClassifier()
        model.booster_ = lgb.Booster(model_file=filepath)
    else:
        # Default to joblib for sklearn and other models
        filepath = f"{input_dir}/{model_name}.pkl"
        model = joblib.load(filepath)

    logger.info(f"Model loaded from: {filepath}")
    return model
