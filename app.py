import pickle
import streamlit as st
import random
from PIL import Image

@st.cache_data()
def load_pickled_objects():
    pickled_vector = pickle.load(open('temp/model/word_vectorDTC90%.pkl', 'rb'))
    pickled_model = pickle.load(open('temp/model/modelDTC90%.pkl', 'rb'))
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
        st.title('Tentang')
        st.markdown("---")
        with st.expander("Aplikasi"):
            st.markdown("""K-Bloom merupakan aplikasi untuk mengklasifikasikan soal berdasarkan level kognitif taksonomi
                        bloom ke dalam dua kelas: HOTS (Higher Order Thinking Skills) dan LOTS (Lower Order Thinking Skills). Aplikasi 
                        ini dikembangkan agar mempermudah guru ataupun calon guru memprediksi kategori soal yang akan digunakan 
                        untuk mengukur pengetahuan peserta didik tentang materi yang bersangkutan.""")
            
        with st.expander("Panduan Penggunaan"):
            st.markdown("1. Masuk ke halaman beranda.")
            st.markdown("2.	Masukan soal dalam bentuk teks berbahasa indonesia.")
            st.markdown("3.	Klik tombol predict.")
            st.markdown("4.	Akan tampil hasil klasifikasi soal berdasarkan level kognitif.")
            
        with st.expander("Metode"):
            st.markdown("""Decision Tree Classifier adalah algoritma pembelajaran mesin yang umumnya digunakan untuk tugas klasifikasi dan regresi. 
                        Algoritma ini bekerja dengan membagi data masukan 
                        secara berulang-ulang menjadi subset berdasarkan nilai fitur-fitur masukan. 
                        Partisi ini direpresentasikan dalam bentuk struktur berupa pohon, di mana setiap simpul internal mewakili suatu keputusan berdasarkan fitur tertentu
                        , dan setiap simpul daun mewakili label kelas (untuk klasifikasi) atau prediksi (untuk regresi).""")
        
        with st.expander("Fitur"):
            st.markdown("- Mengklasifikasikan teks ke dalam dua kategori: HOTS dan LOTS berdasarkan data masukan pengguna.")
            st.markdown("- Menggunakan algoritma Decision Tree Classifier.")
            st.markdown("- Menerapkan TF-IDF sebagai teknik preprocessing untuk merepresentasikan data teks.")
            st.markdown("- Mengeksplorasi hyperparameter untuk mengoptimalkan model klasifikasi.")


def predict_text(text, vectorizer, model):
    
    lots_suggest = [
        'Sebaiknya soal mengandung kemampuan atau keterampilan membedakan, mengorganisasikan, dan menghubungkan. Kata kerja operasional yang biasa digunakan adalah membandingkan, mengkritisi, mengurutkan, membedakan, dan menentukan.',
        'Sebaiknya soal mengandung kata kerja operasional yang digunakan yaitu mengevaluasi, memilih, menilai, menyanggah, dan memberikan pendapat.', 
        'Sebaiknya soal mengandung kemampuan dalam merancang, membangun, merencanakan, memproduksi,  menemukan, dan menyempurnakan. Kata kerja operasional yang digunakan adalah memperjelas, menafsirkan, dan memprediksi.'
    ]
    
    sentence = [text]
    vectorized_text = vectorizer.transform(sentence)
    predict = model.predict(vectorized_text)
    
    # Get the first element (predicted class) from the numpy array
    predicted_class = predict[0]
    
    if predicted_class == 'Lower Order Thinking Skills':
        random_suggestion = random.choice(lots_suggest)
        st.info(f'Teks diprediksi sebagai {predicted_class}.')
        st.info(f'Saran : {random_suggestion}')
    else:
        st.info(f'Teks diprediksi sebagai {predicted_class}.')

if __name__ == '__main__':
    main()