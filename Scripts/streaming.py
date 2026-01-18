from ollama import chat
stream = chat(
    model="llama3.2",
    messages=[{"role": "user", "content": "What is 17 x 23? Please explain your answer."}], # "Please explain your answer." This is a custom add-on."
    stream=True,
)
content_buffer = []
in_thinking = False
for chunk in stream:
    msg = chunk.message
    # Handle thinking (optional)
    if msg.thinking:
        if not in_thinking:
            in_thinking = True
            print("Thinking:\n", end="", flush=True)
        print(msg.thinking, end="", flush=True)
    # Handle final content
    elif msg.content:
        if in_thinking:
            in_thinking = False
            print("\nAnswer:", end=" ", flush=True)
        # Normalize whitespace BEFORE printing
        text = msg.content.replace("\n", " ")
        print(text, end="", flush=True)
        content_buffer.append(text)
print()  # final newline
final_answer = "".join(content_buffer).strip()
