import pickle
import streamlit as st
import os
from PIL import Image

@st.cache_data()
def load_pickled_objects():
    pickled_vector = pickle.load(open('temp/model/word_vectorKNN.pkl', 'rb'))
    pickled_model = pickle.load(open('temp/model/modelKNN.pkl', 'rb'))
    return pickled_vector, pickled_model

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

    st.sidebar.title('Klasifikasi Soal HOTS dan LOTS')
    st.sidebar.image(image_icon)
    choice = st.sidebar.selectbox('Main Menu', ['Beranda', 'Tentang'])
    st.sidebar.info('Web ini dapat melakukan fungsi klasifikasi teks ke dalam kategori HOTS dan LOTS.')

    # if choice == 'Profil':
        
    #     col1, col2, col3 = st.columns(3)
        
    #     # Path gambar
    #     image_path = 'temp/icon/Logo UNY.png'

    #     # Ukuran baru untuk gambar (misal: 300x300 piksel)
    #     new_image_width = 300

    #     with col2:
    #         # Menampilkan gambar dengan ukuran yang berbeda
    #         st.image(image_path, width=new_image_width)
    #     with st.container():
    #         st.markdown('<center><h1>ANALISIS PERBANDINGAN KLASIFIKASI SOAL BERDASARKAN LEVEL KOGNITTIF MENGGUNAKAN METODE DECISION TREE DAN K-NEAREST NEIGHBOORS</h1></center>', unsafe_allow_html=True)
    #         st.header("Nama : Hida Syifaurohmah")
    #         st.header("NIM : 21520251040")
    #         st.header("Pembimbing : Dr. Ir. Fatchul Arifin, M.T")
        
    #     st.markdown('<center><h1>PENDIDIKAN TEKNIK ELEKTRONIKA DAN INFORMATIKA</h1></center>', unsafe_allow_html=True)
    #     st.markdown('<center><h1>PROGRAM PASCA SARJANA</h1></center>', unsafe_allow_html=True)
    #     st.markdown('<center><h1>UNIVERSITAS NEGERI YOGYAKARTA</h1></center>', unsafe_allow_html=True)
    #     st.markdown('<center><h1>2023</h1></center>', unsafe_allow_html=True)


    if choice == 'Beranda':
        st.header("Klasifikasikan teks anda disini!")

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
                st.warning("Silahkan input teks pada form untuk melakukan klasifikasi!")

    elif choice == 'Tentang':
        st.title('About')
        st.markdown("---")
        st.header('Klasifikasi teks : HOTS and LOTS')
        st.markdown("Proyek Klasifikasi Teks bertujuan untuk mengklasifikasikan teks ke dalam dua kelas: HOTS (Higher Order Thinking Skills) dan LOTS (Lower Order Thinking Skills). Tujuannya adalah untuk mengategorikan teks secara akurat berdasarkan frekuensi kejadian mereka. Proyek ini menggunakan dua algoritma klasifikasi: Pohon Keputusan (Decision Tree Classifier) dan Tetangga Terdekat K (K-Nearest Neighbors atau KNN), dan menerapkan teknik pra pemrosesan TF-IDF (Term Frequency-Inverse Document Frequency).")
        st.markdown("")
        st.header('Fitur')
        st.markdown("- Mengklasifikasikan teks ke dalam dua kategori: HOTS dan LOTS berdasarkan frekuensi kejadian mereka.")
        st.markdown("- Menggunakan algoritma Decision Tree Classifier dan KNN untuk klasifikasi.")
        st.markdown("- Menerapkan TF-IDF sebagai teknik pra pemrosesan untuk merepresentasikan data teks.")
        st.markdown("- Mengeksplorasi penyetelan hiperparameter untuk mengoptimalkan model klasifikasi.")
        st.markdown("")
        st.header('Pengembangan Model')
        st.markdown("Proyek ini melibatkan pengembangan dua model klasifikasi: Decision Tree Classifier dan KNN.")
        st.markdown("- Decision Tree Classifier: Model ini membangun sebuah pohon keputusan berdasarkan fitur-fitur yang diambil dari representasi TF-IDF data teks. Ia membagi data berdasarkan frekuensi kejadian dari sekuensi teks untuk mengklasifikasikannya ke dalam kelas-kelas yang sesuai.")
        st.markdown("- K-Nearest Neighbors (KNN): Model ini menggunakan representasi TF-IDF untuk mengukur kesamaan antara teks input dan contoh-contoh pelatihan. Ia mengklasifikasikan teks dengan mempertimbangkan k tetangga terdekat dalam data pelatihan.")
        st.markdown("Kedua model ini menjalani eksplorasi hiperparameter untuk mencari nilai optimal untuk parameter-parameter seperti kedalaman maksimum, kriteria, jumlah tetangga, dan metrik jarak.")

def predict_text(text, vectorizer, model):
    sentence = [text]
    vectorized_text = vectorizer.transform(sentence)
    predict = model.predict(vectorized_text)
    
    # Get the first element (predicted class) from the numpy array
    predicted_class = predict[0]

    st.info(f'Teks diprediksi sebagai {predicted_class}.')

if __name__ == '__main__':
    main()