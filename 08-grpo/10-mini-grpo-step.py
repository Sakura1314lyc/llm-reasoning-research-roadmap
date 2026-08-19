"""
====1 : 放模型
"""

import re
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
    dtype=torch.bfloat16
).to(device)

model.eval()


"""
=== 2 : reward提取答案
"""

def extract_strict_answer(text):

    match = re.search(
        r"####\s*(-?\d+(?:\.\d+)?)\s*$",
        text.strip()
    )

    if match is None:
        return None

    return match.group(1)


def extract_answer(text):

    strict = extract_strict_answer(text)

    if strict is not None:
        return strict


    boxed = re.findall(
        r"\\boxed\{(-?\d+(?:\.\d+)?)\}",
        text
    )

    if boxed:
        return boxed[-1]


    numbers = re.findall(
        r"-?\d+(?:\.\d+)?",
        text
    )

    if numbers:
        return numbers[-1]

    return None


def accuracy_reward(text, ground_truth):

    pred = extract_answer(text)

    if pred is None:
        return 0.0

    return float(
        float(pred) == float(ground_truth)
    )


def format_reward(text):

    return float(
        extract_strict_answer(text)
        is not None
    )


def reward_func(text, ground_truth):

    acc = accuracy_reward(
        text,
        ground_truth
    )

    fmt = format_reward(text)

    return (
        acc
        +
        0.1 * fmt
    )

"""
再次封装log probability

"""

def get_completion_log_probs(
    model,
    full_ids,
    attention_mask,
    prompt_length
):
    outputs = model(
        input_ids = full_ids,
        attention_mask = attention_mask
    )
   
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
A store has 120 notebooks.
It sells 35% of them on Monday.
On Tuesday, it sells 18 fewer notebooks than it sold on Monday.
How many notebooks remain?
""".strip()

ground_truth = "54"

"""
=== 构造提示词
"""

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


prompt_length = (
    inputs["input_ids"].shape[1]
)

"""
=== Rollout 生成回答
"""

G = 4


with torch.no_grad():

    full_ids = model.generate(
        **inputs,
        do_sample=True,
        temperature=1.0,
        top_p=0.95,
        num_return_sequences=G,
        max_new_tokens=192,
        pad_token_id=tokenizer.eos_token_id
    )

completion_ids = full_ids[:, prompt_length:]

is_eos = (completion_ids == tokenizer.eos_token_id)

completion_length = completion_ids.size(1)

eos_position = torch.full(
    (G, ),
    completion_length,
    device = device,
    dtype = torch.long
)

has_eos = is_eos.any(dim = 1)

eos_position[has_eos] = (is_eos.long().argmax(dim = 1)[has_eos])

positions = torch.arange(
    completion_length,
    device=device
).unsqueeze(0)

completion_mask = (
    positions
    <=
    eos_position.unsqueeze(1)
).float()

prompt_mask = torch.ones(
    G,
    prompt_length,
    device=device
)

full_attention_mask = torch.cat(
    [
        prompt_mask,
        completion_mask
    ],
    dim=1
).long()

completion_texts = tokenizer.batch_decode(
    completion_ids,
    skip_special_tokens=True
)

reward_list = []


for i, text in enumerate(
    completion_texts
):

    reward = reward_func(
        text,
        ground_truth
    )

    reward_list.append(
        reward
    )

    print(
        f"\n===== Rollout {i + 1} ====="
    )

    print(text)

    print(
        "Extracted:",
        extract_answer(text)
    )

    print(
        "Reward:",
        reward
    )
rewards = torch.tensor(
    reward_list,
    dtype=torch.float32,
    device=device
)


print(
    "\nRewards:",
    rewards
)

mean_reward = rewards.mean()


std_reward = rewards.std(
    unbiased=False
)


advantages = (
    rewards
    -
    mean_reward
) / (
    std_reward
    +
    1e-8
)
print(
    "Mean reward:",
    mean_reward.item()
)

print(
    "Std reward:",
    std_reward.item()
)

print(
    "Advantages:",
    advantages
)

if std_reward.item() < 1e-8:

    print(
        "\nAll rewards are equal."
    )

    print(
        "This prompt provides no "
        "GRPO learning signal."
    )

    raise SystemExit

advantages = advantages.unsqueeze(1)
with torch.no_grad():

    old_log_probs = (
        get_completion_log_probs(
            model,
            full_ids,
            full_attention_mask,
            prompt_length
        )
    )

old_log_probs = (
    old_log_probs.detach()
)
ref_log_probs = (
    old_log_probs.clone()
)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-5,
    weight_decay=0.0
)
epsilon = 0.2
beta = 0.02

num_policy_updates = 3
for update in range(
    num_policy_updates
):

    current_log_probs = (
        get_completion_log_probs(
            model,
            full_ids,
            full_attention_mask,
            prompt_length
        )
    )
    ratio = torch.exp(
        current_log_probs
        -
        old_log_probs
    )
    objective_1 = (
        ratio
        *
        advantages
    )
    clipped_ratio = torch.clamp(
        ratio,
        1 - epsilon,
        1 + epsilon
    )


    objective_2 = (
        clipped_ratio
        *
        advantages
    )
    per_token_objective = (
        torch.min(
            objective_1,
            objective_2
        )
    )
    token_counts = (
        completion_mask.sum(dim=1)
        .clamp_min(1.0)
    )


    per_sequence_policy_loss = -(
        (
            per_token_objective
            *
            completion_mask
        ).sum(dim=1)
        /
        token_counts
    )


    policy_loss = (
        per_sequence_policy_loss
        .mean()
    )
    ref_log_ratio = (
        ref_log_probs
        -
        current_log_probs
    )


    per_token_kl = (
        torch.exp(
            ref_log_ratio
        )
        -
        ref_log_ratio
        -
        1
    )
    per_sequence_kl = (
        (
            per_token_kl
            *
            completion_mask
        ).sum(dim=1)
        /
        token_counts
    )


    kl_loss = (
        per_sequence_kl.mean()
    )
    loss = (
        policy_loss
        +
        beta * kl_loss
    )
    optimizer.zero_grad(
        set_to_none=True
    )

    loss.backward()

    optimizer.step()
    with torch.no_grad():

        after_log_probs = (
            get_completion_log_probs(
                model,
                full_ids,
                full_attention_mask,
                prompt_length
            )
        )


        after_ratio = torch.exp(
            after_log_probs
            -
            old_log_probs
        )
        sequence_ratios = (
            (
                after_ratio
                *
                completion_mask
            ).sum(dim=1)
            /
            token_counts
        )
        outside_clip = (
            (
                after_ratio
                <
                1 - epsilon
            )
            |
            (
                after_ratio
                >
                1 + epsilon
            )
        )


        clip_fraction = (
            (
                outside_clip.float()
                *
                completion_mask
            ).sum()
            /
            completion_mask.sum()
        )
    print(
        f"\n===== Update {update} ====="
    )

    print(
        "Policy loss:",
        policy_loss.item()
    )

    print(
        "KL loss:",
        kl_loss.item()
    )

    print(
        "Total loss:",
        loss.item()
    )

    print(
        "Mean ratio per rollout:",
        sequence_ratios
    )

    print(
        "Clip fraction:",
        clip_fraction.item()
    )