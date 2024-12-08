import openai
import os
from transformers import pipeline


def generate_response_openai(conversation_history, model="gpt-3.5-turbo"):

    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=conversation_history,
        )
        return response.choices[0].message['content']
    except Exception as e:
        return f"Error: {str(e)}"


def generate_response_huggingface(conversation_history, model="facebook/blenderbot-400M-distill"):

    try:
        hf_pipeline = pipeline("text2text-generation", model=model)
        conversation_text = "\n".join([msg['content'] for msg in conversation_history if msg['role'] == 'user'])
        response = hf_pipeline(conversation_text, max_length=200, truncation=True)
        return response[0]['generated_text']
    except Exception as e:
        return f"Error: {str(e)}"


def main():
    """
    Main function to run the chatbot interface.
    """
    openai.api_key = os.getenv("OPENAI_API_KEY")

    print("Welcome to the PythonChatBotFall24! Type 'exit' to end the chat.")
    print("Select chatbot version: 1 for OpenAI, 2 for Hugging Face")

    bot_choice = input("Your choice: ")

    if bot_choice not in ["1", "2"]:
        print("Invalid choice. Exiting...")
        return

    conversation_history = []

    while True:
        user_input = input("You: ")

        if user_input.lower() == 'exit':
            print("Chatbot: Goodbye! Have a great day!")
            break

        if user_input.lower() == 'switch to openai' and bot_choice == "2":
            print("Switching to OpenAI model...")
            bot_choice = "1"
            continue

        if user_input.lower() == 'switch to huggingface' and bot_choice == "1":
            print("Switching to Hugging Face model...")
            bot_choice = "2"
            continue

        conversation_history.append({"role": "user", "content": user_input})

        if bot_choice == "1":
            response = generate_response_openai(conversation_history)
        else:
            response = generate_response_huggingface(conversation_history)

        conversation_history.append({"role": "assistant", "content": response})

        print(f"Chatbot: {response}")


if __name__ == "__main__":
    main()