from data_loader import load_data

X_eeg, X_eye, y = load_data()

print("EEG array shape:", X_eeg.shape)
print("Eye array shape:", X_eye.shape)
print("Label array shape:", y.shape)
