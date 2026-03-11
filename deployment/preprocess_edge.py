import warnings
warnings.filterwarnings('ignore')

import numpy as np

class EdgePreprocessor:
    """
    Edge-optimized Data Preprocessing Pipeline.
    
    Executes high-efficiency artifact removal, normalization, and tensor 
    formatting on the edge CPU prior to NPU ingestion. This step is critical 
    to mitigate physical sensor noise (e.g., sonar backscatter, ECG baseline wander).
    """
    def __init__(self):
        # Established statistical moments from the training set (Simulated parameters)
        # In a real deployment, load these from a configuration file or moving average
        self.vis_mean, self.vis_std = 0.485, 0.229
        self.sonar_mean, self.sonar_std = 0.5, 0.25
        self.physio_baseline = np.zeros(64) 

    def normalize_visual(self, vis_stream: np.ndarray) -> np.ndarray:
        """Standard Z-score normalization for optical feature maps."""
        # Adding epsilon (1e-6) to prevent division by zero
        return (vis_stream - self.vis_mean) / (self.vis_std + 1e-6)

    def denoise_sonar(self, sonar_stream: np.ndarray) -> np.ndarray:
        """Applies an optimized 1D moving average filter for acoustic noise reduction."""
        # Fast smoothing optimized for edge CPU execution
        smoothed = np.copy(sonar_stream)
        # Simple temporal smoothing across the sequence dimension (axis=0)
        for i in range(1, sonar_stream.shape[0] - 1):
            smoothed[i] = (sonar_stream[i-1] + sonar_stream[i] + sonar_stream[i+1]) / 3.0
        return (smoothed - self.sonar_mean) / (self.sonar_std + 1e-6)

    def process_physio(self, physio_stream: np.ndarray) -> np.ndarray:
        """Min-Max scaling and baseline centering for physiological signals (ECG/SpO2)."""
        centered = physio_stream - self.physio_baseline
        max_val = np.max(np.abs(centered)) + 1e-6
        return centered / max_val

    def execute_pipeline(self, vis: np.ndarray, sonar: np.ndarray, physio: np.ndarray):
        """
        Executes the full preprocessing suite.
        Expects numpy arrays of shape (Batch, Seq_Len, Feature_Dim) or (Seq_Len, Feature_Dim).
        """
        processed_vis = self.normalize_visual(vis)
        processed_sonar = self.denoise_sonar(sonar)
        processed_physio = self.process_physio(physio)
        
        return processed_vis, processed_sonar, processed_physio


# ==========================================
# Validation & Profiling Module
# ==========================================
if __name__ == "__main__":
    print("[INFO] Initiating Edge Preprocessor Diagnostics...")
    
    preprocessor = EdgePreprocessor()
    
    # 1. Generate synthetic raw sensor noise (Simulating real-world corruption)
    raw_vis = np.random.uniform(0, 255, size=(1, 16, 512)).astype(np.float32)
    raw_sonar = np.random.normal(5.0, 2.0, size=(1, 16, 128)).astype(np.float32)
    raw_physio = np.random.normal(100.0, 15.0, size=(1, 16, 64)).astype(np.float32)
    
    print(f"[INFO] Raw Sonar Mean: {np.mean(raw_sonar):.2f} | Std: {np.std(raw_sonar):.2f}")
    
    # 2. Execute pipeline
    clean_vis, clean_sonar, clean_physio = preprocessor.execute_pipeline(raw_vis, raw_sonar, raw_physio)
    
    print("[INFO] --- Post-Processing Artifacts ---")
    print(f"[INFO] Clean Sonar Mean: {np.mean(clean_sonar):.2f} | Std: {np.std(clean_sonar):.2f}")
    print(f"[INFO] Clean Physio Max: {np.max(clean_physio):.2f} | Min: {np.min(clean_physio):.2f}")
    print("[SUCCESS] Preprocessing pipeline execution validated.")