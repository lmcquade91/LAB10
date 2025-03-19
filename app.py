import os
import streamlit as st
import joblib
import pandas as pd
import os
import sys

# Ensure the script can find utils.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import preprocessor  # ✅ This ensures it is imported correctly

def run():
    # Load the trained sentiment analysis model from the same directory
    model_path = os.path.join(os.path.dirname(__file__), "model.joblib")
    model = joblib.load(model_path)  # ✅ Correct path

    st.title("Sentiment Analysis")
    st.text("Basic app to detect the sentiment of text.")
    st.text("")

    userinput = st.text_input('Enter text below, then click the Predict button.', placeholder='Input text HERE')
    st.text("")

    if st.button("Predict"):
        if userinput.strip():  # Ensure input is not empty or just spaces
            try:
                # Convert user input into a Pandas DataFrame (Ensures it has the right format)
                input_series = pd.Series([userinput])  # ✅ Ensure input is a Pandas Series
                
                # Create an instance of the preprocessor and apply transformation
                preprocessor_instance = preprocessor()  # ✅ Fix: Create an instance of the class
                processed_text_series = preprocessor_instance.transform(input_series)  # ✅ Apply transformation
                
                # Ensure processed_text_series is a Pandas Series
                if isinstance(processed_text_series, pd.Series):
                    processed_text = processed_text_series.iloc[0]  # ✅ Extract processed text
                else:
                    processed_text = str(processed_text_series)  # ✅ Convert to string if necessary
                
                # Ensure input to model.predict() is a list
                prediction_input = [processed_text]  # ✅ Convert to list format
                
                # Make a prediction
                predicted_sentiment = model.predict(prediction_input)[0]  # ✅ Ensure proper input format

                output = 'positive 👍' if predicted_sentiment == 1 else 'negative 👎'
                sentiment = f'Predicted sentiment of \"{userinput}\" is {output}.'
                st.success(sentiment)

            except Exception as e:
                st.error(f"An error occurred: {e}")  # ✅ Display a user-friendly error message

if __name__ == "__main__":
    run()
