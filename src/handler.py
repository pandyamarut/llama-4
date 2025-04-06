import runpod
import torch
import os
from transformers import AutoProcessor, Llama4ForConditionalGeneration
from huggingface_hub import login

# Login to Hugging Face using environment variable
hf_token = os.getenv("HF_TOKEN")
login(token=hf_token)

# Load the model into memory before server starts
model_id = "meta-llama/Llama-4-Maverick-17B-128E-Instruct"
processor = AutoProcessor.from_pretrained(model_id)
model = Llama4ForConditionalGeneration.from_pretrained(
    model_id,
    attn_implementation="flex_attention",
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

# Handler function for RunPod
def handler(job):
    job_input = job['input']
    messages = job_input.get("messages")
    stream = job_input.get("stream", False)
    max_new_tokens = job_input.get("max_new_tokens", 256)

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    if stream:
        # Streaming output
        def stream_output():
            streamer = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                streamer=True
            )
            for token in streamer:
                decoded_token = processor.decode(token)
                yield decoded_token

        return stream_output()

    else:
        # Non-streaming output
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
        response = processor.batch_decode(outputs[:, inputs["input_ids"].shape[-1]:])[0]
        return response

# Start the RunPod serverless endpoint
runpod.serverless.start({"handler": handler})
