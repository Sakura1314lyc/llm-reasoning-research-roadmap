import torch.nn.functional as F
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
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

G = 4

"""
generate()不需要梯度, 他的目的是采样action
"""

with torch.no_grad(): 
    gen_outputs = model.generate(
        **inputs,
        do_sample = True,
        temperature = 0.9,
        top_p = 0.95,
        num_return_sequences = G,
        max_new_tokens = 64,
        pad_token_id = tokenizer.eos_token_id
    )
full_ids = gen_outputs
# model.train()

# model是需要梯度的
forward_outputs = model(input_ids = full_ids) 

logits = forward_outputs.logits

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

prompt_length = inputs["input_ids"].shape[1]
completion_log_probs = token_log_probs[
    :,
    prompt_length - 1:
]

advantage = torch.tensor(
    -1.0,
    device = device
)

loss = -(advantage * completion_log_probs).mean()
# print("loss:", loss.item())

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr = 1e-6
)

before = (
    completion_log_probs.detach().mean().item()
)

loss.backward()
optimizer.step()

with torch.no_grad():

    new_outputs = model(input_ids = full_ids)

    new_logits = new_outputs.logits[:, :-1, :]

    new_log_probs = F.log_softmax(new_logits, dim = -1)

    new_token_log_probs = torch.gather(
        new_log_probs,
        dim = - 1,
        index = shift_labels.unsqueeze(-1)
    ).squeeze(-1)

    new_completion_log_probs = (
        new_token_log_probs[:, prompt_length - 1:]
    )
after = (
    new_completion_log_probs.mean().item()
)

print(f"before: {before:.10f}")
print(f"after: {after:.10f}")
print(f"diff : {after-before:.10f}")

# prob_ratio = torch.exp(new_log_probs - log_probs)
# uncilpped = (prob_ratio * advantage)

# epsilon = 1e-6

# clipped_ratio = torch.clamp(
#     prob_ratio,
#     1 - epsilon,
#     1 + epsilon
# )

# clipped = (clipped_ratio * advantage)

# Loss = -torch.min(uncilpped, clipped).mean()

# print("ratio : ", prob_ratio)
# print("Loss : ", Loss)