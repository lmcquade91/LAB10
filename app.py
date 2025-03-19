import os
import joblib

def run():
    # Load the trained sentiment analysis model from the same directory
    model_path = os.path.join(os.path.dirname(__file__), "model.joblib")
    model = joblib.load(model_path)  # ✅ Correct path

    st.title("Sentiment Analysis")
    st.text("Basic app to detect the sentiment of text.")
    st.text("")

    userinput = st.text_input('Enter text below, then click the Predict button.', placeholder='Input text HERE')
    st.text("")

    predicted_sentiment = ""

    if st.button("Predict"):
        processed_text = preprocessor(userinput)
        predicted_sentiment = model.predict([processed_text])[0]

        output = 'positive 👍' if predicted_sentiment == 1 else 'negative 👎'
        sentiment = f'Predicted sentiment of \"{userinput}\" is {output}.'
        st.success(sentiment)

if __name__ == "__main__":
    run()
