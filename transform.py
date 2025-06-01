from keras.models import model_from_json, Sequential  # <-- Explicitly import Sequential
import h5py

# Load architecture
with h5py.File("asl_classifier.h5", "r") as f:
    model_json = f.attrs["model_config"]

# Deserialize
model = model_from_json(model_json)

# Load weights
model.load_weights("asl_classifier1.h5")