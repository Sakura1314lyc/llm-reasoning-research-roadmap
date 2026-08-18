import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    dtype = torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype = torch.bfloat16
)

model = model.to(device)
model.eval()
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
    tokenize = False,
    add_generation_prompt = True
)

inputs = tokenizer(
    prompt,
    return_tensors = "pt"
).to(device)

print(prompt)
print(
    "input_ids shape : ",
    inputs["input_ids"].shape
)
G = 4
with torch.inference_mode():
    outputs = model.generate(
        **inputs,
        do_sample = True,
        temperature = 0.9,
        top_p = 0.95,
        num_return_sequences = G,
        max_new_tokens = 256,
        pad_token_id = tokenizer.eos_token_id
    )

prompt_length = inputs["input_ids"].shape[1]

for i, outputs_ids in enumerate(outputs):

    completion_ids = outputs_ids[
        prompt_length:
    ]

    completion = tokenizer.decode(
        completion_ids,
        skip_special_tokens = True
    )

    print(
        f"\n ==== Rollout {i + 1}==="
    )

    print(completion)