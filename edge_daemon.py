import warnings
warnings.filterwarnings('ignore')

import os
import sys
import time
import numpy as np

# Ensure absolute imports work from the project root
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 1. Import Perception & Alignment Layers (Real Device Entry Points)
from data.alignment import SensorAPIClient, MultiModalStreamAligner

# 2. Import Preprocessing Layer
from deployment.preprocess_edge import EdgePreprocessor

# 3. Import Decision & Action Layers
from agents.qwen_engine import QwenReasoningAdapter, DRMAPPO_Scheduler

# 4. Import Inference Layer (with graceful degradation for testing environments)
try:
    from deployment.acl_inference import AscendEdgeInferencer
    HAS_MSLITE = True
except ImportError:
    HAS_MSLITE = False
    class MockInferencer:
        """Fallback mock inferencer to allow pipeline testing on non-NPU environments."""
        def predict(self, vis, sonar, physio):
            # Simulate a 35ms NPU inference latency
            time.sleep(0.035) 
            # Randomly simulate distress probabilities for demonstration
            # Inject a high probability randomly to trigger the LLM sequence
            is_distress = np.random.rand() > 0.8 
            prob = [0.1, 0.9] if is_distress else [0.95, 0.05]
            return np.array([prob]), 35.0

class DrowningDetectionDaemon:
    """
    Main Edge Runtime Daemon.
    
    Orchestrates the continuous "Perception -> Preprocessing -> Inference -> 
    Decision -> Action" loop on the deployment hardware. Operates via a 
    non-blocking while loop to guarantee real-time telemetry processing.
    """
    def __init__(self, target_freq_hz: int = 10, distress_threshold: float = 0.85):
        self.target_freq = target_freq_hz
        self.threshold = distress_threshold
        
        print("="*60)
        print("[SYSTEM] Booting Drowning Detection Edge Daemon...")
        print("="*60)
        
        # Initialize Subsystems
        self._init_perception()
        self._init_inference()
        self._init_cognitive_engine()
        
        print("[SYSTEM] All subsystems initialized. Daemon ready.")
        print("="*60)

    def _init_perception(self):
        print("[INIT] Connecting to Hardware Sensor APIs...")
        # Using the real API client defined in data/alignment.py
        self.api_client = SensorAPIClient(base_url="http://127.0.0.1:8080/api/v1")
        self.aligner = MultiModalStreamAligner(
            api_client=self.api_client, 
            target_freq_hz=self.target_freq, 
            window_size_sec=1.6
        )
        self.preprocessor = EdgePreprocessor()

    def _init_inference(self):
        print("[INIT] Loading CADT Perceptual Core...")
        if HAS_MSLITE:
            self.inferencer = AscendEdgeInferencer(mindir_path="Drowning-Detection-System/deployment/cadt_edge_model.mindir", device_id=0)
        else:
            print("[WARNING] NPU Drivers (mslite) not detected. Operating in Mock Inference Mode.")
            self.inferencer = MockInferencer()

    def _init_cognitive_engine(self):
        print("[INIT] Loading Qwen Cognitive Engine and DR-MAPPO Swarm Scheduler...")
        self.llm_adapter = QwenReasoningAdapter()
        self.swarm_scheduler = DRMAPPO_Scheduler(num_usvs=2, num_uuvs=1)

    def run_loop(self):
        """
        The continuous execution loop. 
        Runs continuously until terminated via SIGINT (Ctrl+C).
        """
        print("\n[SYSTEM] Entering Real-Time Observation Loop (Press Ctrl+C to terminate)...\n")
        cycle_count = 0
        
        try:
            while True:
                cycle_count += 1
                loop_start = time.time()
                
                # Step 1: Data Ingestion & Alignment (Real API Pull)
                # fetch_and_align returns MindSpore Tensors
                t_vis, t_sonar, t_physio = self.aligner.fetch_and_align()
                
                # Convert to numpy for preprocessing and edge inference compatibility
                np_vis = t_vis.asnumpy()
                np_sonar = t_sonar.asnumpy()
                np_physio = t_physio.asnumpy()
                
                # Step 2: Edge Preprocessing (Noise reduction)
                clean_vis, clean_sonar, clean_physio = self.preprocessor.execute_pipeline(np_vis, np_sonar, np_physio)
                clean_vis = clean_vis.astype(np.float32)
                clean_sonar = clean_sonar.astype(np.float32)
                clean_physio = clean_physio.astype(np.float32)
                
                # Step 3: NPU Inference
                probs, latency_ms = self.inferencer.predict(clean_vis, clean_sonar, clean_physio)
                distress_prob = float(probs[0][1]) # Index 1 corresponds to 'Distress' class
                
                # Console telemetry output
                status = "NORMAL" if distress_prob < self.threshold else "CRITICAL"
                print(f"[Cycle {cycle_count:04d}] Inference: {latency_ms:.1f}ms | Distress Prob: {distress_prob:.3f} | Status: {status}")
                
                # ==========================================
                # Step 4: Cognitive Decision Trigger & UI Telemetry
                # ==========================================
                qwen_strategy_payload = None
                swarm_status_payload = None

                if distress_prob >= self.threshold:
                    print(f"\n[ALERT] Threshold exceeded ({distress_prob:.2f} >= {self.threshold}). Awakening Cognitive Engine...")
                    
                    # Extract raw physiological vector to pass to LLM
                    physio_vector_snapshot = clean_physio[0, 0, :2] 
                    
                    # LLM Semantic Reasoning
                    strategy = self.llm_adapter.generate_strategy(distress_prob, physio_vector_snapshot)
                    qwen_strategy_payload = strategy  # 捕获策略用于前端展示
                    
                    # Swarm Execution
                    print("[ACTION] Dispatching Swarm Directives:")
                    actions = self.swarm_scheduler.dispatch_agents(strategy, target_coordinates=(34.21, 118.45))
                    
                    if actions:
                        swarm_status_payload = actions[0] # 捕获第一条指令用于前端弹窗
                        
                    for action in actions:
                        print(f"   -> {action}")
                    print("-" * 60)
                
                # ------------------------------------------
                # 【新增核心功能】前端大屏联动推送 (Telemetry Push)
                # ------------------------------------------
                # 注意：必须使用 float() 强制转换 Numpy 数据类型，否则会引发 JSON 序列化崩溃
                current_hrv = float(clean_physio[0, 0, 0])
                
                telemetry_data = {
                    "distress_prob": float(distress_prob),
                    "physio_hrv": current_hrv,
                    "latency_ms": float(latency_ms),
                    "qwen_strategy": qwen_strategy_payload,
                    "swarm_status": swarm_status_payload
                }
                
                try:
                    import requests 
                    requests.post("http://127.0.0.1:8000/api/push_telemetry", json=telemetry_data, timeout=0.5)
                except Exception as e:
                    print(f"[WARNING] 无法将数据推送至大屏: {e}")

                # ==========================================
                # Step 5: Enforce loop frequency
                # ==========================================
                elapsed = time.time() - loop_start
                sleep_time = max(0.0, (1.0 / self.target_freq) - elapsed)
                time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\n[SYSTEM] SIGINT received. Executing graceful shutdown sequence...")
            print("[SYSTEM] Halting sensor streams and parking multi-agent swarm.")
            print("[SYSTEM] Daemon terminated successfully.")


if __name__ == '__main__':
    # Instantiate and run the daemon
    daemon = DrowningDetectionDaemon(target_freq_hz=10, distress_threshold=0.85)
    daemon.run_loop()