from orchestrator.prefect_flow import travel_chatbot_flow


def main():
    print("Thai Travel Chatbot (type 'exit' to quit)")
    while True:
        question = input("\nถามได้เลย: ").strip()
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            print("bye")
            break

        answer = travel_chatbot_flow(question)
        print("\n" + answer)


if __name__ == "__main__":
    main()
