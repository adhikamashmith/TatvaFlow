# Import libraries
import pickle
import joblib
import io

# Input sentence
sentence = "I am Ashmit"

# -------------------------------
# Using Pickle
# -------------------------------
pickle_bytes = pickle.dumps(sentence)  # serialize
print("Pickle Serialized Byte Stream:")
print(pickle_bytes)
print("\n")

# -------------------------------
# Using Joblib
# -------------------------------
# Joblib usually writes to a file, but we can use BytesIO to get byte stream
bytes_io = io.BytesIO()
joblib.dump(sentence, bytes_io)
joblib_bytes = bytes_io.getvalue()
print("Joblib Serialized Byte Stream:")
print(joblib_bytes)
print("\n")