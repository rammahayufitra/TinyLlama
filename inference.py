from transformers import GenerationConfig
from time import perf_counter

def generate_response(user_input):

  prompt = formatted_prompt(user_input)

  inputs = tokenizer([prompt], return_tensors="pt")
  generation_config = GenerationConfig(penalty_alpha=0.6,do_sample = True,
      top_k=5,temperature=0.5,repetition_penalty=1.2,
      max_new_tokens=12,pad_token_id=tokenizer.eos_token_id
  )
  start_time = perf_counter()

  inputs = tokenizer(prompt, return_tensors="pt").to('cuda')

  outputs = model.generate(**inputs, generation_config=generation_config)
  print(tokenizer.decode(outputs[0], skip_special_tokens=True))
  output_time = perf_counter() - start_time
  print(f"Time taken for inference: {round(output_time,2)} seconds")

def formatted_prompt(question)-> str:
    return f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant:"

def print_color_space(hex_color):
    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    r, g, b = hex_to_rgb(hex_color)
    print(f'{hex_color}: \033[48;2;{r};{g};{b}m           \033[0m')


generate_response(user_input='Light Orange color')


print_color_space('#f0901a ')