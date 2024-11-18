import gzip
import pickle

# Load your model
with open("traffic_mapie.pickle", "rb") as f:
    model = pickle.load(f)

# Save with compression
with gzip.open("traffic_mapie_compressed.pickle.gz", "wb") as f:
    pickle.dump(model, f)
print("Compressed and saved as traffic_mapie_compressed.pickle.gz")
