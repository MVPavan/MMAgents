import model_explorer
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torchinfo import summary


torch.set_grad_enabled(False)
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
cache_dir = '/media/data_2/vlm/hf_models'

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    # torch_dtype=torch.bfloat16,
    device_map="cpu",
    cache_dir=cache_dir,
    trust_remote_code=True
).eval()
tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir,trust_remote_code=True)
messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]

input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)

# terminators = [
#     tokenizer.eos_token_id,
#     tokenizer.convert_tokens_to_ids("<|eot_id|>")
# ]

# outputs = model.generate(
#     input_ids,
#     max_new_tokens=256,
#     eos_token_id=terminators,
#     do_sample=True,
#     temperature=0.6,
#     top_p=0.9,
# )
# response = outputs[0][input_ids.shape[-1]:]
# print(tokenizer.decode(response, skip_special_tokens=True))

inputs = (model.dummy_inputs['input_ids'],)
print("Model summary ...")
summary(model, input_data = inputs, device='cpu', depth=3)

# print("Exporting Model ...")
# ep = torch.export.export(model, inputs)
# print("Visualizing Model ...")
# model_explorer.visualize_pytorch('llama3', exported_program=ep)