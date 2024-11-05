import time
import openai
import os

openai.api_key = os.environ.get("BACKWARD_OPENAI_KEY")


def call_lm(
    prompt,
    model="gpt-3.5-turbo-0125",
    temperature=0,
    max_tokens=128,
    logprobs=None,
    stop=None,
    system_prompt="You are a helpful Assistant. You will follow the format of the prompt from the user and complete the rest.",
):
    if "instruct" in model:
        while 1:
            try:
                response = openai.Completion.create(
                    model=model,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stop=stop,
                    logprobs=logprobs,
                )
                break
            except:
                print("Openai API error, sleep for 1s")
                time.sleep(1)
        return response.choices[0].text, response.choices[0].logprobs
    else:
        while 1:
            try:
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stop=stop,
                )
                break
            except:
                print("Openai API error, sleep for 1s")
                time.sleep(1)
        return response.choices[0].message.content.strip(), None


if __name__ == "__main__":
    print(call_lm("hello"))
