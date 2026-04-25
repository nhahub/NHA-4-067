import streamlit as st
import pickle
import numpy as np

# load model + encoder
model = pickle.load(open("F:\project\Intelligent-Support-Ticket\Project Implementation\Classifications_models\Issue_typeModel\Issue_type.pkl", "rb"))
label_encoder = pickle.load(open("F:\project\Intelligent-Support-Ticket\Project Implementation\Classifications_models\Issue_typeModel\label_encoder.pkl", "rb"))

st.title("🎫 Support Ticket Classifier")

user_input = st.text_area("Enter your issue:")

# حنفّكر المتغير هنا
top3_predictions = []

def process_top3(pred_probs):
    """
    فانكشن تستخدمها بعد كده لأي حاجه 
    مثلا توصيات، logging، أو تربطها بحل
    """
    top3_idx = np.argsort(pred_probs)[-3:][::-1]
    top3_labels = label_encoder.inverse_transform(top3_idx)
    top3_scores = pred_probs[top3_idx]
    # نرجع قائمة tuples (label, score)
    return list(zip(top3_labels, top3_scores))

if st.button("🔍 Predict") and user_input.strip() != "":
    text = user_input.lower()
    pred = model.predict([text])
    label = label_encoder.inverse_transform(pred)
    
    # get probabilities
    probs = model.predict_proba([text])[0]
    
    # هنا نحفظ Top 3 في الفيلد اللي هنستدعيه بعد كده
    top3_predictions = process_top3(probs)
    
    # عرض النتيجة في UI
    st.success(f"✅ Issue Type: {label[0]}")
    st.info(f"🎯 Confidence: {np.max(probs):.2f}")

    st.markdown("### 🔝 Top 3 Predictions")
    for i, (l, score) in enumerate(top3_predictions):
        st.write(f"{i+1}. {l} ({score:.2f})")

# مثال على استخدام top3_predictions في فانكشن تاني
def recommend_solution():
    if top3_predictions:
        st.write("💡 Suggested solution for top prediction:")
        st.write(f"Solution for {top3_predictions[0][0]} goes here...")
    else:
        st.write("No prediction yet!")

recommend_solution()