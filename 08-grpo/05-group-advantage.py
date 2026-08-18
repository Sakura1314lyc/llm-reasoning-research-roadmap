import torch

rewards = torch.tensor([
    1.1,
    0.0,
    1.0,
    0.1
])

mean_reward = rewards.mean()

std_reward = rewards.std(
    unbiased=False
)

advantages = (rewards - mean_reward) / (std_reward + 1e-8)

print("Rewards:", rewards)
print("\nMean reward:", mean_reward)
print("\nStd reward:", std_reward)
print("\nAdvantages:", advantages)