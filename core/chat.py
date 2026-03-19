import os
import Path
from dotenv import load_dotenv
from rag import ask
from pathlib import Path

load_dotenv(Path(__file__).parent.parent / ".env")

def main():
    print("🤖 Personal Assistant Chatette is ready!")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["quit", "exit"]:
                print("Goodbye!")
                break

            ask(user_input)

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break

if __name__ == "__main__":
    main()