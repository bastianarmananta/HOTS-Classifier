import pickle
import streamlit as st
from PIL import Image

@st.cache_data()
def load_pickled_objects():
    pickled_vector = pickle.load(open('temp/model/vector_newDataset.pkl', 'rb'))
    pickled_model = pickle.load(open('temp/model/model_newDataset.pkl', 'rb'))
    return pickled_vector, pickled_model

def load_sample_text():
    with open('temp/text/sample.txt', 'r') as file:
        sample_text = file.readlines()
    return [text.strip() for text in sample_text]

def main():
    session_state = st.session_state
    if 'selected_text' not in session_state:
        session_state.selected_text = ""
    if 'predict_button_clicked' not in session_state:
        session_state.predict_button_clicked = False

    image_icon = Image.open('temp/icon/icon2.png')
    st.set_page_config(
        page_title='HOTS Classifier',
        layout='wide',
        initial_sidebar_state='auto',
        page_icon=image_icon
    )

    st.sidebar.title('HOTS Classifier')
    st.sidebar.image(image_icon)
    choice = st.sidebar.selectbox('Main Menu', ["Profile", 'Home', 'About'])
    st.sidebar.info('This web app will text classification into HOTS, and LOTS categories.')

    if choice == 'Profile':
        st.title("Name")
        st.title("NIM")
        st.title("etc")

    if choice == 'Home':
        st.title("Classify your text!")
        sample_text = load_sample_text()
        selected_text = st.radio('Select a sample text:', sample_text, key='sample_text')

        if selected_text != session_state.selected_text:
            session_state.selected_text = selected_text
            session_state.predict_button_clicked = True

        text_input = st.text_input(
            label="Input text here",
            placeholder="Pusat pengaturan lalulintas data dalam vsat dinamakan dengan",
            label_visibility='hidden',
            value=session_state.selected_text
        )

        predict_button = st.button('Predict', key='predict', type='primary')

        if predict_button or session_state.predict_button_clicked:
            session_state.predict_button_clicked = False

            if text_input:
                pickled_vector, pickled_model = load_pickled_objects()
                predict_text(text_input, pickled_vector, pickled_model)
            else:
                st.warning("Please input your text to classify")

    elif choice == 'About':
        st.title('About')
        st.markdown("---")
        st.header('Text Classification: HOTS and LOTS')
        st.markdown("The Text Classification project aims to classify text into three classes: HOTS (Highly Occurring Text Sequences) and LOTS (Low Occurring Text Sequences). The goal is to accurately categorize text based on their occurrence frequency. The project utilizes two classification algorithms: Decision Tree Classifier and K-Nearest Neighbors (KNN), and applies TF-IDF (Term Frequency-Inverse Document Frequency) as a preprocessing technique.")
        st.markdown("")
        st.header('Features')
        st.markdown("- Classify text into three categories: HOTS,  and LOTS based on their occurrence frequency.")
        st.markdown("- Utilize Decision Tree Classifier and KNN algorithms for classification.")
        st.markdown("- Apply TF-IDF as a preprocessing technique to represent text data.")
        st.markdown("- Explore hyperparameter tuning to optimize the classification models.")
        st.markdown("")
        st.header('Model Development')
        st.markdown("The project involves the development of two classification models: Decision Tree Classifier and KNN.")
        st.markdown("- Decision Tree Classifier: This model builds a decision tree based on the features derived from the TF-IDF representation of the text data. It splits the data based on the occurrence frequency of text sequences to classify them into the respective classes.")
        st.markdown("- K-Nearest Neighbors (KNN): This model utilizes the TF-IDF representation to measure the similarity between the input text and the training instances. It classifies the text by considering the k nearest neighbors in the training data.")
        st.markdown("Both models undergo hyperparameter exploration to find the optimal values for parameters such as maximum depth, criterion, number of neighbors, and distance metric.")

def predict_text(text, vectorizer, model):
    sentence = [text]
    vectorized_text = vectorizer.transform(sentence)
    predict = model.predict(vectorized_text)

    st.info(f'Text predicted as {predict}')

if __name__ == '__main__':
    main()