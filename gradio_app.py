import gradio as gr
import requests
import os
from dotenv import load_dotenv

# Config
load_dotenv()
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/ask")
TIMEOUT = 60  # reranker can be slow on CPU

# API call
def query_rag(question: str) -> str:
    question = question.strip()
    if not question:
        return "Please enter a question."
    
    try:
        response = requests.post(
            API_URL,
            json={"question": question},
            timeout=TIMEOUT
        )

        if response.status_code == 503:
            return "The assistant is currently unavailable. Please ensure the API server is running."

        if response.status_code == 422:
            return "Invalid request — your question may be too long or empty."

        response.raise_for_status()

        data = response.json()
        answer = data.get("answer", "No answer returned.")
        sources = data.get("sources", [])

        # Format sources nicely
        if sources:
            sources_text = "\n".join(f"- {s}" for s in sources)
            answer += f"\n\n**Sources:**\n{sources_text}"

        return answer

    except requests.exceptions.ConnectionError:
        return "Cannot connect to the API server. Please make sure it is running (`uvicorn api:app`)."

    except requests.exceptions.Timeout:
        return "The request timed out. The pipeline may be overloaded — please try again."

    except Exception as e:
        return f"Unexpected error occurred: {e}"


# Chat handler
def chat_response(message, history):
    history = history or []

    if not message.strip():
        return history, ""

    # Detect language for display
    arabic_chars = sum(1 for c in message if "\u0600" <= c <= "\u06FF")
    lang_tag = "🇱🇧 Arabic" if arabic_chars > len(message) * 0.3 else "🇬🇧 English"

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": f"*Detected: {lang_tag} — processing...*"})

    answer = query_rag(message)

    history[-1] = {"role": "assistant", "content": answer}

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
        Ask questions about **Employment Rights** and **Self-Employment** law in Lebanon.
        Supports both **English** and **Arabic** (عربي).
        """
    )
    # Layout: Chat + Sidebar
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="Legal Assistant",
                height=420
            )

            with gr.Row():
                user_input = gr.Textbox(
                    placeholder="Type your question in English or Arabic... / اكتب سؤالك بالعربية أو الإنجليزية",
                    show_label=False,
                    lines=1,
                    scale=5
                )
                send_button = gr.Button("Ask ➤", scale=1, variant="primary")

            clear_button = gr.Button("🗑 Clear Chat", variant="secondary")

        # Sidebar column 
        with gr.Column(scale=1):
            gr.Markdown(
                """
                ### How to use:
                1. Type your legal question in English or Arabic.
                2. Press **Ask** or hit **Enter**.
                3. The assistant answers strictly from the official guides.
                4. Sources are shown below each answer.

                ---

                ### Example questions
                - *What are the basic rights of an employee?*
                - *كيف تزداد الإجازة السنوية بناءً على سنوات الخدمة؟*
                - *Is maternity leave paid and for how long?*
                - *هل يحق للعمال الأجانب الضمان الاجتماعي؟*

                ---

                ### Note
                The assistant will explicitly say if a question is **not covered** by the guides.
                It does not speculate or hallucinate.

                ---
                *Powered by multilingual E5 embeddings, BM25, cross-encoder reranking, and GPT-4.1-mini*
                """
            )

    # Connect components
    send_button.click(
        chat_response, 
        inputs=[user_input, chatbot], 
        outputs=[chatbot, user_input]
        )
    user_input.submit(
        chat_response, 
        inputs=[user_input, chatbot], 
        outputs=[chatbot, user_input]
        )
    clear_button.click(clear_chat, outputs=[chatbot])


# Launch
if __name__ == "__main__":
    app.launch(share=False, theme=gr.themes.Soft())
