import gradio as gr

from rag_assistant.qa import KnowledgeAssistant

assistant = None


def respond(message, history):
    global assistant
    try:
        if assistant is None:
            assistant = KnowledgeAssistant()

        answer, sources = assistant.answer(message)

        if sources:
            answer += "\n\n**Retrieved sources**\n" + "\n".join(
                f"- **[{s['label']}] {s['title']}**" + (f" (page ID: {s['page_id']})" if s["page_id"] else "")
                for s in sources
            )

        return answer
    except Exception as exc:
        return (
            "The assistant could not complete the request. Check the index and "
            "Azure OpenAI configuration.\n\n"
            f"Technical detail: `{type(exc).__name__}: {exc}`"
        )


demo = gr.ChatInterface(
    fn=respond,
    title="Knowledge Assistant",
    description=(
        "Ask questions about the indexed documentation. Verify important "
        "information against the retrieved source pages."
    ),
    examples=[
        "What is the purpose of this documentation?",
        "Which steps are described in the process?",
    ],
)

if __name__ == "__main__":
    demo.launch()
