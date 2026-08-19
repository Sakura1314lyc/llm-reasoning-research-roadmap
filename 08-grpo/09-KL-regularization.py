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
    ref_log_probs = old_log_probs.clone()

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
beta = 0.05
completion_ids = full_ids[:, prompt_length:]

is_eos = (
    completion_ids
    ==
    tokenizer.eos_token_id
)

eos_idx = torch.full(
    (completion_ids.size(0),),
    completion_ids.size(1),
    device=device,
    dtype=torch.long
)

has_eos = is_eos.any(dim=1)

eos_idx[has_eos] = (
    is_eos.int()
    .argmax(dim=1)[has_eos]
)

positions = torch.arange(
    completion_ids.size(1),
    device=device
).unsqueeze(0)

completion_mask = (
    positions
    <= eos_idx.unsqueeze(1)
).float()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr = 1e-5,
    weight_decay=0.0
)

beta = 0.05
epsilon = 0.2

for step in range(5):

    current_log_probs = get_completion_log_probs(
        model,
        full_ids,
        prompt_length
    )


    # -------------------------
    # 1. PPO / GRPO ratio
    # -------------------------

    ratio = torch.exp(
        current_log_probs
        -
        old_log_probs
    )


    unclipped_objective = (
        ratio
        *
        advantages
    )


    clipped_ratio = torch.clamp(
        ratio,
        1 - epsilon,
        1 + epsilon
    )


    clipped_objective = (
        clipped_ratio
        *
        advantages
    )


    objective = torch.min(
        unclipped_objective,
        clipped_objective
    )


    policy_loss = -(
        objective
        *
        completion_mask
    ).sum() / completion_mask.sum()


    # -------------------------
    # 2. KL penalty
    # -------------------------

    log_ratio_ref = (
        ref_log_probs
        -
        current_log_probs
    )


    per_token_kl = (
        torch.exp(log_ratio_ref)
        -
        log_ratio_ref
        -
        1
    )


    kl_loss = (
        per_token_kl
        *
        completion_mask
    ).sum() / completion_mask.sum()


    # -------------------------
    # 3. Final loss
    # -------------------------

    loss = (
        policy_loss
        +
        beta * kl_loss
    )


    optimizer.zero_grad()

    loss.backward()

    optimizer.step()


    print(
        f"\nstep={step}"
    )

    print(
        "policy loss:",
        policy_loss.item()
    )

    print(
        "KL:",
        kl_loss.item()
    )

    print(
        "total loss:",
        loss.item()
    )