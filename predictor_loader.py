# predictor_loader.py
import os
import json
import joblib
import h5py
import numpy as np
import pandas as pd
import traceback

try:
    from tensorflow.keras.models import load_model, model_from_json
except Exception:
    from keras.models import load_model, model_from_json


def _fix_model_config(config_dict):
    """Recursively fix legacy config by removing batch_shape and replacing with batch_input_shape."""
    if isinstance(config_dict, dict):
        if "config" in config_dict and isinstance(config_dict["config"], dict):
            cfg = config_dict["config"]
            if "batch_shape" in cfg:
                # replace with batch_input_shape (TF2 expects this)
                if "batch_input_shape" not in cfg:
                    cfg["batch_input_shape"] = cfg.pop("batch_shape")
                else:
                    cfg.pop("batch_shape", None)
        for k, v in list(config_dict.items()):
            _fix_model_config(v)
    elif isinstance(config_dict, list):
        for item in config_dict:
            _fix_model_config(item)


class EnrollmentPredictor:
    def __init__(self, model, feature_scaler=None, target_scaler=None, encoder=None, time_steps=1, feature_names=None):
        self.model = model
        self.feature_scaler = feature_scaler
        self.target_scaler = target_scaler
        self.encoder = encoder
        self.time_steps = time_steps
        self.feature_names = feature_names

    def preprocess_features(self, features):
        if isinstance(features, dict):
            df = pd.DataFrame([features])
        else:
            df = pd.DataFrame(features)

        if self.feature_names:
            for fn in self.feature_names:
                if fn not in df.columns:
                    df[fn] = 0
            df = df[self.feature_names]

        if self.feature_scaler is not None:
            try:
                scaled = self.feature_scaler.transform(df)
            except Exception:
                scaled = np.asarray(df, dtype=float)
        else:
            scaled = np.asarray(df, dtype=float)

        if scaled.ndim == 1:
            scaled = scaled.reshape(-1, 1)

        if len(scaled) < self.time_steps:
            if len(scaled) == 0:
                padded = np.zeros((self.time_steps, scaled.shape[1]))
            else:
                last_row = scaled[-1]
                padding = np.tile(last_row, (self.time_steps - len(scaled), 1))
                padded = np.vstack([scaled, padding])
            scaled = padded
        elif len(scaled) > self.time_steps:
            scaled = scaled[-self.time_steps:]

        return scaled.reshape((1, self.time_steps, scaled.shape[1]))

    def predict(self, features):
        X = self.preprocess_features(features)
        y_pred = self.model.predict(X, verbose=0)
        if self.target_scaler is not None:
            try:
                y_inv = self.target_scaler.inverse_transform(y_pred)
                val = float(np.ravel(y_inv)[0])
            except Exception:
                val = float(np.ravel(y_pred)[0])
        else:
            val = float(np.ravel(y_pred)[0])
        return val

    def forecast(self, history, steps=5):
        if self.target_scaler is not None:
            hist = np.array(history).reshape(-1, 1)
            hist_scaled = self.target_scaler.transform(hist)
            input_seq = hist_scaled[-self.time_steps:]
            forecasted = []
            for _ in range(steps):
                X = input_seq.reshape((1, self.time_steps, 1))
                y_pred = self.model.predict(X, verbose=0)
                y_inv = self.target_scaler.inverse_transform(y_pred)[0][0]
                forecasted.append(float(y_inv))
                input_seq = np.vstack([input_seq[1:], y_pred.reshape(1, -1)])
            return forecasted
        else:
            seq = list(history[-self.time_steps:])
            forecasted = []
            for _ in range(steps):
                X = np.array(seq).reshape((1, self.time_steps, 1))
                y_pred = self.model.predict(X, verbose=0)
                val = float(np.ravel(y_pred)[0])
                forecasted.append(val)
                seq = seq[1:] + [val]
            return forecasted


def safe_load_model(path):
    """
    Attempts to load a model safely.
    - Supports new `.keras` format
    - Fixes legacy `.h5` files with `batch_shape` issue
    """
    try:
        model = load_model(path, compile=False)
        print(f"✅ load_model succeeded for {path}")
        return model
    except Exception as e:
        print(f"⚠️ Standard load failed: {e}")

    # Legacy H5 fix
    if path.endswith(".h5"):
        try:
            with h5py.File(path, "r") as f:
                model_config_attr = None
                if "model_config" in f.attrs:
                    model_config_attr = f.attrs["model_config"]
                elif "model_config" in f.keys():
                    model_config_attr = f["model_config"][()]

                if model_config_attr is None:
                    raise ValueError("No model_config found in HDF5 file")

                # Handle bytes, numpy.string_, or str
                if isinstance(model_config_attr, (bytes, bytearray)):
                    config_str = model_config_attr.decode("utf-8")
                elif hasattr(model_config_attr, "tobytes"):  # numpy.string_
                    config_str = model_config_attr.tobytes().decode("utf-8")
                else:
                    config_str = str(model_config_attr)

                config = json.loads(config_str)

                # Fix batch_shape → batch_input_shape
                def _fix_config(cfg):
                    if isinstance(cfg, dict):
                        if "batch_shape" in cfg:
                            if "batch_input_shape" not in cfg:
                                cfg["batch_input_shape"] = cfg.pop("batch_shape")
                            else:
                                cfg.pop("batch_shape", None)
                        for v in cfg.values():
                            _fix_config(v)
                    elif isinstance(cfg, list):
                        for v in cfg:
                            _fix_config(v)

                _fix_config(config)

            # Rebuild model
            model_json = json.dumps(config)
            model = model_from_json(model_json)
            model.load_weights(path)

            # Save a fixed `.keras` copy for future
            fixed_path = path.replace(".h5", "_fixed.keras")
            try:
                model.save(fixed_path)
                print(f"💾 Saved fixed model as {fixed_path}")
            except Exception as e_save:
                print(f"⚠️ Could not save .keras copy: {e_save}")

            print("✅ Legacy .h5 model loaded successfully (batch_shape fixed)")
            return model

        except Exception as e2:
            print("❌ Legacy loader failed:", e2)
            traceback.print_exc()
            return None

    return None
 

def load_enrollment_predictor(models_dir="models"):
    try:
        keras_path = os.path.join(models_dir, "enrollment_predictor.keras")
        h5_path = os.path.join(models_dir, "enrollment_predictor_fixed.h5")

        model_path = None
        if os.path.exists(keras_path):
            model_path = keras_path
        elif os.path.exists(h5_path):
            model_path = h5_path
        else:
            print("❌ No model file found in", models_dir)
            return None

        print("📂 Loading model from", model_path)
        model = safe_load_model(model_path)
        if model is None:
            print("❌ Model load returned None")
            return None

        feature_scaler = None
        target_scaler = None
        encoder = None
        feature_names = None
        time_steps = 1

        if os.path.exists(os.path.join(models_dir, "feature_scaler.pkl")):
            try:
                feature_scaler = joblib.load(os.path.join(models_dir, "feature_scaler.pkl"))
            except Exception as e:
                print("⚠️ Could not load feature_scaler.pkl:", e)

        if os.path.exists(os.path.join(models_dir, "target_scaler.pkl")):
            try:
                target_scaler = joblib.load(os.path.join(models_dir, "target_scaler.pkl"))
            except Exception as e:
                print("⚠️ Could not load target_scaler.pkl:", e)

        if os.path.exists(os.path.join(models_dir, "encoder.pkl")):
            try:
                encoder = joblib.load(os.path.join(models_dir, "encoder.pkl"))
            except Exception as e:
                print("⚠️ Could not load encoder.pkl:", e)

        if os.path.exists(os.path.join(models_dir, "feature_names.json")):
            try:
                with open(os.path.join(models_dir, "feature_names.json"), "r", encoding="utf-8") as fh:
                    feature_names = json.load(fh)
            except Exception as e:
                print("⚠️ Could not load feature_names.json:", e)

        if os.path.exists(os.path.join(models_dir, "time_steps_info.json")):
            try:
                with open(os.path.join(models_dir, "time_steps_info.json"), "r", encoding="utf-8") as fh:
                    ts_info = json.load(fh)
                    time_steps = int(ts_info.get("time_steps", 1))
            except Exception as e:
                print("⚠️ Could not load time_steps_info.json:", e)

        predictor = EnrollmentPredictor(
            model=model,
            feature_scaler=feature_scaler,
            target_scaler=target_scaler,
            encoder=encoder,
            time_steps=time_steps,
            feature_names=feature_names,
        )

        print("✅ EnrollmentPredictor instance created")
        return predictor

    except Exception as e:
        print("❌ Error in load_enrollment_predictor:", e)
        traceback.print_exc()
        return None


if __name__ == "__main__":
    p = load_enrollment_predictor()
    if p:
        print("Loaded predictor OK")
    else:
        print("Failed to load predictor")
