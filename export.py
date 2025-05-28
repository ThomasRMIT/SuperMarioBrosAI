import torch
import torch.nn as nn
from stable_baselines3 import PPO

# === Custom Wrapper to Expose Policy + Value Outputs ===
class PolicyValueWrapper(nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, obs):
        # Feature extraction
        features = self.policy.extract_features(obs)

        # Separate latent features for policy and value heads
        latent_pi, latent_vf = self.policy.mlp_extractor(features)

        # Action distribution logits
        distribution = self.policy._get_action_dist_from_latent(latent_pi)
        action_logits = distribution.distribution.logits

        # Value network output
        value = self.policy.value_net(latent_vf)

        return action_logits, value

# === Main Export Logic ===
if __name__ == "__main__":
    # === Configuration ===
    MODEL_PATH = "ppo_mario_checkpoint_1m_1.zip"
    OUTPUT_ONNX_PATH = "ppo_mario_policy_value.onnx"

    # === Load Model ===
    print(f"🔄 Loading PPO model from: {MODEL_PATH}")
    model = PPO.load(MODEL_PATH)
    obs_shape = model.observation_space.shape
    print(f"📐 Expected input shape: {obs_shape}")

    # === Create Wrapper for ONNX Export ===
    wrapped_model = PolicyValueWrapper(model.policy).to(model.device)
    wrapped_model.eval()

    # === Create Dummy Input ===
    dummy_input = torch.zeros((1, *obs_shape)).to(model.device)

    # === Export to ONNX ===
    print(f"📤 Exporting ONNX model to: {OUTPUT_ONNX_PATH}")
    torch.onnx.export(
        wrapped_model,
        dummy_input,
        OUTPUT_ONNX_PATH,
        input_names=["observation"],
        output_names=["action_logits", "value"],
        opset_version=11,
        do_constant_folding=True,
        export_params=True
    )

    print(f"✅ Export complete! File saved at: {OUTPUT_ONNX_PATH}")
    print("👉 Open this in Netron: https://netron.app")