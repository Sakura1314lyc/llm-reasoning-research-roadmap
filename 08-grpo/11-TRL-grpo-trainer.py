import re
import torch

from datasets import Dataset
from peft import LoraConfig

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from trl import (
    GRPOConfig,
    GRPOTrainer,
)

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

tokenizer.padding_side = "left"

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

raw_data = [
    {
        "question": (
            "A store has 120 notebooks. "
            "It sells 35% of them on Monday. "
            "On Tuesday, it sells 18 fewer notebooks "
            "than it sold on Monday. "
            "How many notebooks remain?"
        ),
        "ground_truth": "54",
    },
    {
        "question": (
            "A farmer has 84 eggs. "
            "He sells 3/7 of them in the morning "
            "and then sells 12 more in the afternoon. "
            "How many eggs remain?"
        ),
        "ground_truth": "36",
    },
    {
        "question": (
            "A class has 40 students. "
            "65% are girls. "
            "How many boys are there?"
        ),
        "ground_truth": "14",
    },
    {
        "question": (
            "Sarah had 96 dollars. "
            "She spent one fourth of it on food "
            "and 18 dollars on a book. "
            "How much money remains?"
        ),
        "ground_truth": "54",
    },
]

SYSTEM_PROMPT = (
    "Solve the math problem step by step. "
    "At the end, output the final numerical answer "
    "in exactly this format: #### <answer>"
)

data = []

for item in raw_data:

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": item["question"],
        },
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    data.append(
        {
            "prompt": prompt,
            "ground_truth": item["ground_truth"],
        }
    )

dataset = Dataset.from_list(data)
def extract_answer(text):

    strict = re.findall(
        r"####\s*(-?\d+(?:\.\d+)?)",
        text
    )

    if strict:
        return strict[-1]

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

def accuracy_reward(
    completions,
    ground_truth,
    **kwargs,
):
    rewards = []

    for completion, truth in zip(
        completions,
        ground_truth,
    ):
        pred = extract_answer(
            completion
        )

        if pred is None:
            rewards.append(0.0)
            continue

        try:
            correct = (
                float(pred)
                ==
                float(truth)
            )
        except ValueError:
            correct = False

        rewards.append(
            1.0 if correct else 0.0
        )

    return rewards
def format_reward(
    completions,
    **kwargs,
):

    rewards = []

    pattern = (
        r"####\s*"
        r"-?\d+(?:\.\d+)?"
        r"\s*$"
    )

    for completion in completions:

        match = re.search(
            pattern,
            completion.strip()
        )

        rewards.append(
            1.0 if match else 0.0
        )

    return rewards

peft_config = LoraConfig(
    r=8,
    lora_alpha=16,

    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ],

    lora_dropout=0.05,

    bias="none",

    task_type="CAUSAL_LM",
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16,
)

training_args = GRPOConfig(

    output_dir=(
        "outputs/"
        "qwen25-05b-grpo-pilot"
    ),

    # ----------------
    # training
    # ----------------

    max_steps=5,

    learning_rate=1e-5,

    per_device_train_batch_size=1,

    gradient_accumulation_steps=4,

    bf16=True,

    gradient_checkpointing=True,

    # ----------------
    # rollout
    # ----------------

    num_generations=4, # G

    max_completion_length=192, 

    mask_truncated_completions=True,
    
    temperature=1.0,

    top_p=0.95,

    # ----------------
    # GRPO
    # ----------------

    scale_rewards="group", #自动计算advantage

    num_iterations=2, #每批的generation做几轮 policy update

    epsilon=0.2,

    beta=0.0,

    loss_type="grpo",

    reward_weights=[
        1.0,
        0.1,
    ],

    # ----------------
    # log
    # ----------------

    logging_steps=1,

    logging_first_step=True,

    log_completions=True,

    num_completions_to_print=4,

    report_to="none",

    save_strategy="no",
)

trainer = GRPOTrainer(

    model=model,

    args=training_args,

    train_dataset=dataset,

    reward_funcs=[
        accuracy_reward,
        format_reward,
    ],

    processing_class=tokenizer,

    peft_config=peft_config,
)
trainer.train()
