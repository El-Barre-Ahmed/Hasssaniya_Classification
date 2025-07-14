import joblib
import gradio as gr
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "model_and_vectorizer.pkl")

data = joblib.load(MODEL_PATH)
model = data['model']
vectorizer = data['vectorizer']

def predict_with_polarity(text):
    X = vectorizer.transform([text])
    result = model.predict(X)[0]
    probs = model.predict_proba(X)[0]
    pos, neu, neg = probs
    polarity_score = max(probs)
    return result, f"{polarity_score:.2f}", pos, neu, neg

def predict_and_display(comment):
    if not comment.strip():
        raise gr.Error("⚠️ الرجاء إدخال نص بالحسانية / Veuillez entrer un texte en hassaniya.")
    if len(comment.strip().split()) < 3:
        raise gr.Error("⚠️ يجب إدخال 3 كلمات على الأقل / Veuillez entrer au moins 3 mots.")

    prediction, polarity_rate, pos, neu, neg = predict_with_polarity(comment)
    probs = f"إيجابي: {pos:.2f} | محايد: {neu:.2f} | سلبي: {neg:.2f}"
    return prediction, polarity_rate, probs

with gr.Blocks(title="Analysis Sentiment") as iface:
    gr.Markdown("""<p>
    🗣️ تحليل المشاعر في النصوص الحسانية / Analyse de sentiment - Hassaniya
    """)

    with gr.Row():
        input_box = gr.Textbox(
            label="📝 النص الحساني / Texte Hassaniya",
            placeholder="أدخل نصاً بالحسانية هنا...",
            lines=4
        )

    with gr.Row():
        submit_btn = gr.Button("إرسال / Envoyer")
        reset_btn = gr.Button("إعادة تعيين / Reset")

    with gr.Column():
        output_pred = gr.Textbox(label=" التصنيف / Prédiction", interactive=False)
        output_polarity = gr.Textbox(label=" معدل الاستقطاب / Taux de polarité", interactive=False)
        output_probs = gr.Textbox(label=" النسب / Probabilités", interactive=False)

    submit_btn.click(
        fn=predict_and_display,
        inputs=input_box,
        outputs=[output_pred, output_polarity, output_probs]
    )

    reset_btn.click(
        fn=lambda: ("", "", ""),
        inputs=None,
        outputs=[output_pred, output_polarity, output_probs]
    )

iface.launch(True)
