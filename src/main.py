from vllm import LLM, SamplingParams


def main():

    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    llm = LLM(model='Qwen/Qwen3-0.6B')
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
    outputs = llm.generate(prompts, sampling_params)
    for output in outputs:
        print('output: ', output)
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)
