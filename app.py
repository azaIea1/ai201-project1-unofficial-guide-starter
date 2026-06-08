"""
app.py — Gradio web interface for the Unofficial BU Housing Guide RAG system.

Run with:
    python app.py

Then open http://localhost:7860 in your browser.

Dependencies:
    gradio>=6.9.0  (add to requirements.txt and pip install)
"""

import gradio as gr
from query import ask


def handle_query(question: str) -> tuple[str, str]:
    """Called by Gradio on every button click / Enter press."""
    question = question.strip()
    if not question:
        return "Please enter a question.", ""

    result = ask(question)
    answer = result["answer"]

    # Format sources as a bullet list
    sources_text = "\n".join(f"• {s}" for s in result["sources"])

    # Append retrieval quality info for transparency
    chunk_info = "\n\nRetrieval details (top chunks):\n"
    for i, c in enumerate(result["chunks"], 1):
        chunk_info += f"  [{i}] {c['source']}  (distance: {c['distance']})\n"

    return answer, sources_text + chunk_info


# ---------------------------------------------------------------------------
# UI layout
# ---------------------------------------------------------------------------

with gr.Blocks(title="BU Unofficial Housing Guide") as demo:
    gr.Markdown(
        """
        # 🏠 BU Unofficial Housing Guide
        Ask questions about off-campus housing near Binghamton University.
        Answers are grounded in student Reddit discussions — no hallucination.
        """
    )

    with gr.Row():
        with gr.Column(scale=2):
            question_box = gr.Textbox(
                label="Your question",
                placeholder="e.g. When should I start looking for off-campus housing?",
                lines=2,
            )
            ask_btn = gr.Button("Ask", variant="primary")

        with gr.Column(scale=3):
            answer_box = gr.Textbox(
                label="Answer",
                lines=10,
                interactive=False,
            )
            sources_box = gr.Textbox(
                label="Retrieved from",
                lines=6,
                interactive=False,
            )

    # Example queries
    gr.Examples(
        examples=[
            ["When should I start looking for off-campus housing near BU?"],
            ["Is it cheaper to live off campus than on campus at Binghamton?"],
            ["What are common problems students have with off-campus roommates?"],
            ["What do students say about the experience of living off campus for the first time?"],
            ["How do full-time students afford off-campus housing without working full-time?"],
            ["What landlords or property management companies should I avoid?"],
        ],
        inputs=question_box,
    )

    # Wire up interactions
    ask_btn.click(handle_query, inputs=question_box, outputs=[answer_box, sources_box])
    question_box.submit(handle_query, inputs=question_box, outputs=[answer_box, sources_box])


if __name__ == "__main__":
    demo.launch()
