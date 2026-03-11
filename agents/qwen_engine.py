import warnings
warnings.filterwarnings('ignore')

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import json
import numpy as np
import mindspore as ms

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
except ImportError:
    raise ImportError("[ERROR] Please install required libraries: pip install transformers torch accelerate")

class QwenReasoningAdapter:
    """
     execution adapter for the Qwen Large Language Model (LLM).
    
    This class handles the automated downloading, local caching, and actual 
    inference generation of the Qwen model. It bridges the semantic gap between 
    the numerical probabilities from the CADT perceptual layer and the discrete 
    action space of the downstream RL swarm scheduler.
    """
    def __init__(self, model_repo="Qwen/Qwen2.5-0.5B-Instruct"):
        """
        Initializes the LLM Adapter and ensures weights are cached locally.
        
        Args:
            model_repo: The HuggingFace repository ID for the Qwen model.
        """
        self.model_repo = model_repo
        
        # Define the local cache directory relative to this script
        # Path: Drowning-Detection-System/models/llm_cache
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.cache_dir = os.path.abspath(os.path.join(current_dir, "..", "model_weights", "llm_cache"))
        
        print(f"[INFO] LLM Cache Directory configured at: {self.cache_dir}")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self._initialize_llm_backend()

    def _initialize_llm_backend(self):
        """
        Downloads (if not cached) and loads the Qwen tokenizer and model into memory.
        Uses 'auto' device mapping to leverage available GPU/CPU resources.
        """
        print(f"[INFO] Loading Model: {self.model_repo}. This may take a while during the first download...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_repo, 
            cache_dir=self.cache_dir,
            trust_remote_code=True
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_repo, 
            cache_dir=self.cache_dir,
            device_map="auto",
            torch_dtype="auto", 
            trust_remote_code=True
        )
        
        self.model.eval()
        print("[INFO] Qwen LLM Backend successfully initialized and cached.")

    def _construct_messages(self, distress_prob, physiological_status, environmental_context):
        """
        Constructs the structured conversation format required by Qwen-Instruct.
        """
        system_prompt = (
            "You are an autonomous aquatic rescue commander. Analyze the sensory data "
            "and output ONLY a JSON execution plan. Do not include markdown formatting "
            "or explanatory text. Format: "
            '{"severity_level": "low|medium|critical", "primary_action": "action_name", "requires_aed": true/false}'
        )
        
        user_prompt = (
            f"Sensory Input:\n"
            f"- Distress Probability: {distress_prob:.2f}\n"
            f"- Physiological Status: {physiological_status}\n"
            f"- Environmental Context: {environmental_context}\n"
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        return messages

    def generate_strategy(self, distress_prob, physio_vector):
        """
        Executes  inference on the Qwen model to generate the rescue strategy.
        """
        # 1. Translate physiological vector to semantic description
        hrv_anomaly = physio_vector[0] > 0.8
        physio_status = "Cardiac Arrest Suspected" if hrv_anomaly else "Fatigue / Struggling"
        
        # 2. Construct Chat Messages
        messages = self._construct_messages(distress_prob, physio_status, "High waves, low visibility")
        
        # 3. Apply Chat Template and Tokenize
        text = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        # 4. Generate Output via LLM
        # print("[INFO] Executing Qwen inference...")
        generated_ids = self.model.generate(
            **model_inputs, 
            max_new_tokens=128,
            temperature=0.1, # Low temperature for deterministic JSON output
            do_sample=False
        )
        
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # 5. Parse JSON strictly
        try:
            # Clean up potential markdown formatting (e.g., ```json ... ```)
            clean_text = response_text.replace("```json", "").replace("```", "").strip()
            strategy_dict = json.loads(clean_text)
            return strategy_dict
        except json.JSONDecodeError as e:
            print(f"[ERROR] LLM returned invalid JSON: {response_text}")
            # Fallback safety protocol
            return {"severity_level": "critical", "primary_action": "human_intervention_required", "requires_aed": False}


class DRMAPPO_Scheduler:
    """
    Distributed Recurrent Multi-Agent Proximal Policy Optimization (DR-MAPPO) Engine.
    Translates high-level semantics into continuous kinematic waypoints.
    """
    def __init__(self, num_usvs=2, num_uuvs=1):
        self.num_usvs = num_usvs
        self.num_uuvs = num_uuvs
        
    def dispatch_agents(self, strategy_dict, target_coordinates):
        execution_log = []
        severity = strategy_dict.get("severity_level", "low")
        
        if severity == "critical":
            execution_log.append(f"[USV-1] Dispatched to {target_coordinates} at MAX speed.")
            if strategy_dict.get("requires_aed", False):
                execution_log.append(f"[USV-1] Initiating AED payload drop sequence.")
            execution_log.append(f"[UUV-1] Submerging for underwater visual verification.")
            
        elif severity == "medium":
            execution_log.append(f"[USV-1] Approaching {target_coordinates} for close observation.")
            execution_log.append(f"[USV-2] Establishing communication relay perimeter.")
            
        else:
            execution_log.append("[Swarm] Maintaining standard patrol formation.")
            
        return execution_log


# ==========================================
# Integration & Profiling Module
# ==========================================
if __name__ == '__main__':
    print("[INFO] Initializing  Qwen-DRMAPPO Collaborative Decision Engine...")
    
    # 1. Initialize Modules (This will trigger the download if not cached)
    qwen_adapter = QwenReasoningAdapter(model_repo="Qwen/Qwen2.5-0.5B-Instruct")
    swarm_scheduler = DRMAPPO_Scheduler(num_usvs=2, num_uuvs=1)
    
    # 2. Simulated perceptual input
    mock_cadt_distress_prob = 0.92 
    mock_physio_features = np.array([0.85, 0.4]) 
    mock_target_location = (34.21, 118.45)
    
    print("\n[INFO] --- Perceptual Trigger Received ---")
    print(f"Distress Probability: {mock_cadt_distress_prob:.2f}")
    
    # 3. Semantic Reasoning ( LLM Forward Pass)
    print("\n[INFO] --- Executing  Qwen Semantic Reasoning ---")
    strategy = qwen_adapter.generate_strategy(mock_cadt_distress_prob, mock_physio_features)
    print(f"Computed Strategy: {json.dumps(strategy, indent=2)}")
    
    # 4. Swarm Dispatch via DR-MAPPO
    print("\n[INFO] --- DR-MAPPO Swarm Execution ---")
    actions = swarm_scheduler.dispatch_agents(strategy, mock_target_location)
    for action in actions:
        print(action)