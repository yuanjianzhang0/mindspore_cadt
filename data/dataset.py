import warnings
warnings.filterwarnings('ignore')

import numpy as np
import mindspore.dataset as ds
from typing import Tuple, List

class MultiModalDistressDataset:
    """
    Offline Multi-Modal Dataset Generator for CADT Training.
    
    This class simulates the loading of temporally aligned historical data 
    (Visual, Sonar, Physiological) and their corresponding drowning/normal labels.
    In a real scenario, this would read from HDF5, TFRecord, or MindRecord files.
    """
    def __init__(self, num_samples: int = 1000, seq_len: int = 16):
        """
        Args:
            num_samples: Total number of simulated training samples.
            seq_len: The temporal sequence length (e.g., 16 frames).
        """
        self.num_samples = num_samples
        self.seq_len = seq_len
        
        # Simulated dataset generation (Replace with actual disk I/O in production)
        # Visual: [N, 16, 512], Sonar: [N, 16, 128], Physio: [N, 16, 64]
        self.data_vis = np.random.randn(num_samples, seq_len, 512).astype(np.float32)
        self.data_sonar = np.random.randn(num_samples, seq_len, 128).astype(np.float32)
        self.data_physio = np.random.randn(num_samples, seq_len, 64).astype(np.float32)
        
        # Labels: 0 for Normal Swimming, 1 for Aquatic Distress / Drowning
        # Ensuring a balanced dataset distribution for simulation
        self.labels = np.random.randint(0, 2, size=(num_samples,)).astype(np.int32)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.int32]:
        """
        Retrieves a single synchronized multi-modal sample and its label.
        """
        return (
            self.data_vis[index], 
            self.data_sonar[index], 
            self.data_physio[index], 
            self.labels[index]
        )

    def __len__(self) -> int:
        return self.num_samples


def create_cadt_dataloader(batch_size: int = 32, num_samples: int = 1000, is_training: bool = True):
    """
    Constructs the MindSpore dataset pipeline with shuffling and batching.
    
    Args:
        batch_size: Number of samples per gradient update.
        num_samples: Total dataset size.
        is_training: If True, applies data shuffling.
        
    Returns:
        mindspore.dataset.BatchDataset: The iterable dataloader.
    """
    # 1. Instantiate the Python generator
    dataset_generator = MultiModalDistressDataset(num_samples=num_samples)
    
    # 2. Wrap into MindSpore GeneratorDataset
    # Explicitly defining column names to match the CADT network inputs
    column_names = ["vis_x", "sonar_x", "physio_x", "label"]
    
    dataset = ds.GeneratorDataset(
        source=dataset_generator, 
        column_names=column_names, 
        shuffle=is_training
    )
    
    # 3. Apply batching and optional dropping of remainder to maintain static tensor shapes
    dataset = dataset.batch(batch_size, drop_remainder=True)
    
    return dataset


if __name__ == "__main__":
    print("[INFO] Testing Multi-Modal Dataloader Pipeline...")
    dataloader = create_cadt_dataloader(batch_size=4, num_samples=100)
    
    for batch_idx, data in enumerate(dataloader.create_dict_iterator()):
        print(f"[INFO] Batch {batch_idx + 1}")
        print(f"       Visual Shape: {data['vis_x'].shape}")
        print(f"       Sonar Shape:  {data['sonar_x'].shape}")
        print(f"       Physio Shape: {data['physio_x'].shape}")
        print(f"       Labels:       {data['label'].asnumpy()}")
        break # Just test the first batch
    print("[INFO] Dataloader ready for training loop.")