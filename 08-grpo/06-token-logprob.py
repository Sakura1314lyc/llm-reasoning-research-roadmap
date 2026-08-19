## 找一下损失
"""
generate()
= 我采取了什么 action

model(...)
= 这些 action 在当前 policy 下概率是多少( Π(a|p) )

"""
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

# print(prompt)
# print(
#     "input_ids shape : ",
#     inputs["input_ids"].shape
# )
G = 4
with torch.inference_mode():
    gen_outputs = model.generate(
        **inputs,
        do_sample = True,
        temperature = 0.9,
        top_p = 0.95,
        num_return_sequences = G,
        max_new_tokens = 256,
        pad_token_id = tokenizer.eos_token_id
    )
full_ids = gen_outputs[0].unsqueeze(0)
with torch.no_grad():
    forward_outputs = model(
        input_ids = full_ids
    )

logits = forward_outputs.logits

print("full_ids shape: ", full_ids.shape)
print("logits shape : ", logits.shape)

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

print("shift_logits shape : ", shift_logits.shape)
print("shift_labels shape : ", shift_labels.shape)
print("token_log_probs shape : ", token_log_probs.shape)

prompt_lenth = inputs["input_ids"].shape[1]
completion_ids = full_ids[0, prompt_lenth: ]
completion_log_probs = token_log_probs[0, prompt_lenth - 1: ]

print("completion token count:", len(completion_ids))
print("logprob count:", len(completion_log_probs))

for token_id, log_prob in zip(
    completion_ids,
    completion_log_probs
):
    token_text = tokenizer.decode(
        [token_id.item()]
    )

    probability = torch.exp(
        log_prob
    ).item()

    print(
        repr(token_text),
        "log_prob =",
        round(log_prob.item(), 4),
        "prob =",
        round(probability, 4)
    )

sequence_log_prob = (
    completion_log_probs.sum()
)

print(
    "Sequence log probability:",
    sequence_log_prob.item()
)