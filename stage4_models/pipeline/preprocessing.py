from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pipeline.config import PCA_VARIANCE

def preprocess_eeg(X_eeg):
    """
    Applies StandardScaler and PCA to retain 95% variance.
    Returns normalized and PCA reduced array.
    """
    scaler = StandardScaler()
    scaled = scaler.fit_transform(X_eeg)
    
    pca = PCA(n_components=PCA_VARIANCE)
    reduced = pca.fit_transform(scaled)
    
    return reduced, pca, scaler

def preprocess_eye(X_eye):
    """
    Drop redundant indices 5 and 9.
    Apply StandardScaler.
    """
    # Drop indices 5 and 9
    indices_to_keep = [i for i in range(X_eye.shape[1]) if i not in (5, 9)]
    cleaned = X_eye[:, indices_to_keep]
    
    scaler = StandardScaler()
    scaled = scaler.fit_transform(cleaned)
    return scaled, scaler
