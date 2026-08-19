import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)
tokenizer = AutoTokenizer.from_pretrained(
    model_name
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype = torch.bfloat16
).to(device)

model.eval()

def get_completion_log_probs(model, full_ids, prompt_length):
    outputs = model(input_ids = full_ids)

    logits = outputs.logits

    shift_logits = logits[:, :-1, :]
    shift_labels = full_ids[:, 1:]

    log_probs = F.log_softmax(
        shift_logits,
        dim = -1
    )
    token_log_probs = torch.gather(
        log_probs,
        dim = -1,
        index = shift_labels.unsqueeze(-1)
    ).squeeze(-1)

    completion_log_probs = token_log_probs[:, prompt_length - 1:]

    return completion_log_probs

question = """
Janet has 12 apples.
She gives 3 apples to each of 2 friends.
How many apples does she have left?
""".strip()

messages = [
    {
        "role": "system",
        "content": (
            "Solve the math problem step by step. "
            "At the end, output the final numerical answer "
            "in exactly this format: #### <answer>"
        )
    },
    {
        "role": "user",
        "content": question
    }
]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = tokenizer(
    prompt,
    return_tensors="pt"
).to(device)

prompt_length = inputs["input_ids"].shape[1]

G = 4
with torch.no_grad():
    full_ids = model.generate(
        **inputs,
        do_sample = True,
        temperature = 0.9,
        top_p = 0.95,
        num_return_sequences = G,
        max_new_tokens = 64,
        pad_token_id = tokenizer.eos_token_id
    )

with torch.no_grad():
    old_log_probs = get_completion_log_probs(model, full_ids, prompt_length)
    old_log_probs = old_log_probs.detach()

rewards = torch.tensor(
    [
        1.0,
        0.0,
        1.0,
        0.0,
    ],
    device = device
)


mean_reward = rewards.mean()
std_reward = rewards.std(
    unbiased=False
)


advantages = (rewards - mean_reward) / (std_reward + 1e-8)
advantages = advantages.unsqueeze(1)

epsilon = 0.2

completion_ids = full_ids[:, prompt_length:]

completion_mask = (
    completion_ids != tokenizer.eos_token_id
).float()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr = 1e-5,
    weight_decay=0.0
)

for step in range(5):
    current_log_probs = get_completion_log_probs(model, full_ids, prompt_length)

    ratio = torch.exp(current_log_probs - old_log_probs)

    unclipped_objective = (ratio * advantages)

    clipped_ratio = torch.clamp(
        ratio,
        1 - epsilon,
        1 + epsilon
    )
    clipped_objective = (
        clipped_ratio * advantages
    )

    objectvice = torch.min(unclipped_objective, clipped_objective)

    loss = -(completion_mask * objectvice).sum() / completion_mask.sum()

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    with torch.no_grad():

        sequence_ratios = (ratio * completion_mask).sum(dim = 1) / completion_mask.sum(dim = 1)

        print(f"\nstep={step}")

        print("Advantages:", advantages.squeeze(1))

        print("Mean ratios per rollout:", sequence_ratios)
