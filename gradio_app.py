import gradio as gr
import requests

# FastAPI Endpoint
API_URL = "http://127.0.0.1:8000/ask"


# Query function
def query_rag(question):
    try:
        response = requests.post(
            API_URL,
            json={"question": question},
            timeout=20
        )
        response.raise_for_status()

        data = response.json()
        answer = data.get("answer", "No answer found.")
        sources = data.get("sources", [])

        # Format sources nicely
        if sources:
            answer += "\n\nSources:\n"
            for s in sources:
                answer += f"- {s}\n"

        return answer

    except Exception as e:
        return f"Unexpected error occurred: {e}"


# Chat function
def chat_response(message, history):
    history = history or []

    answer = query_rag(message)

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": answer})

    return history, ""


# Clear chat
def clear_chat():
    return []


# UI
with gr.Blocks() as app:
    # Header
    gr.Markdown(
        """
        # 📚 Legal RAG Assistant
        Ask questions about **Employment Rights** and **Self-Employment Legal Guides** in Lebanon.
        """,
        elem_id="header",
    )

    # Layout: Chat + Sidebar
    with gr.Row():
        # Chat column (main)
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(label="Legal Assistant", height=300)

            with gr.Row():
                user_input = gr.Textbox(
                    placeholder="Type your legal question here...",
                    show_label=False,
                    lines=1
                )
                send_button = gr.Button("Ask")

            clear_button = gr.Button("Clear Chat")

        # Sidebar column 
        with gr.Column(scale=1):
            gr.Markdown(
                """
                ### How to use:
                1. Type your legal question below.
                2. Press **Ask** or hit **Enter**.
                3. Wait a few seconds for the assistant to respond.
                4. Sources for answers are shown below each response.
                """
            )
            gr.Markdown(
                """
                *Powered by RAG pipeline & OpenAI API*
                """
            )

    # Connect components
    send_button.click(chat_response, inputs=[user_input, chatbot], outputs=[chatbot, user_input])
    user_input.submit(chat_response, inputs=[user_input, chatbot], outputs=[chatbot, user_input])
    clear_button.click(clear_chat, outputs=[chatbot])

    pass


# Launch
if __name__ == "__main__":
    app.launch(theme=gr.themes.Soft(), share=True)
