import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
from tensorflow.image import resize

# ------------------ Functions ------------------
@st.cache_resource()
def load_model():
    model = tf.keras.models.load_model("./Trained_model.h5")
    return model

def load_and_preprocess_data(file_path, target_shape=(150, 150)):
    data = []
    audio_data, sample_rate = librosa.load(file_path, sr=None)
    chunk_duration = 4
    overlap_duration = 2
    chunk_samples = chunk_duration * sample_rate
    overlap_samples = overlap_duration * sample_rate
    num_chunks = int(np.ceil((len(audio_data) - chunk_samples) / (chunk_samples - overlap_samples))) + 1
    for i in range(num_chunks):
        start = i * (chunk_samples - overlap_samples)
        end = start + chunk_samples
        chunk = audio_data[start:end]
        mel_spectrogram = librosa.feature.melspectrogram(y=chunk, sr=sample_rate)
        mel_spectrogram = resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
        data.append(mel_spectrogram)
    return np.array(data)

def model_prediction(X_test):
    model = load_model()
    y_pred = model.predict(X_test)
    predicted_categories = np.argmax(y_pred, axis=1)
    unique_elements, counts = np.unique(predicted_categories, return_counts=True)
    max_count = np.max(counts)
    max_elements = unique_elements[counts == max_count]
    return max_elements[0]

# ------------------ Page Setup ------------------
st.set_page_config(page_title="Music Genre Classifier", layout="wide")

# Hide Streamlit default menu and footer
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stApp {background-color: -webkit-linear-gradient(to right, #4b134f, #c94b4b); color: white;}
    </style>
""", unsafe_allow_html=True)

# ------------------ Home Section ------------------


st.markdown("""
<div style="text-align:center; padding-top:50px;">
    <h1 style="font-size:3em; animation: fadeIn 2s;">Welcome to My Music <span>Genre</span> Classification app 🎶🎧</h1>
    <p style="font-size:1.2em; max-width:700px; margin:auto; animation: fadeIn 4s; color:#7A7A7A;">
    my goal is to help in identifying music genres from audio tracks efficiently. Upload an audio file, and the app will analyze it to detect its genre. Discover the power of AI in music analysis!
    </p>
</div>

<style>

/* ---------------- GLOBAL APP BACKGROUND ---------------- */
.stApp {
     background: #D3CCE3;  /* fallback for old browsers */
background: -webkit-linear-gradient(to right, #E9E4F0, #D3CCE3);  /* Chrome 10-25, Safari 5.1-6 */
background: linear-gradient(to right, #E9E4F0, #D3CCE3); /* W3C, IE 10+/ Edge, Firefox 16+, Chrome 26+, Opera 12+, Safari 7+ */

    color: white;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
}
            
/* ------------ FULL-WIDTH FIXED FOOTER ------------ */
#custom-footer {
    position: fixed;
    bottom: 0;
    left: 0;
    width: 100%;
    background: linear-gradient(to right, #bc4e9c, #f80759);
    text-align: center;
    padding: 18px;
    color: white;
    font-weight: 600;
    border-top: 2px solid rgba(255,255,255,0.3);
    z-index: 9999;
}
#custom-footer a {
    background: linear-gradient(90deg, #ff9a9e, #fad0c4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 1.1em;
    font-weight: bold;
}


/* -------------- HEADINGS COLOR ---------------- */
h2, h3, .stHeader {
    color: #ff496d; !important;
}
h1{
    color: #fff !important;
    span {
        color: #ff496d !important;}        
            }

/* Streamlit header element specifically */
h1 span, h2 span, h3 span {
    color: #ff496d !important;
}

/* Markdown header override */
section[data-testid="stHeader"] h1 {
    color: #ffce54 !important;
}


/* ------------------- ANIMATIONS (unchanged) ------------------- */
@keyframes fadeIn {
    from {opacity:0;}
    to {opacity:1;}
}
@keyframes slideInLeft {
    from {opacity:0; transform:translateX(-100px);}
    to {opacity:1; transform:translateX(0);}
}
@keyframes bounce {
    0%, 100% {transform: translateY(0);}
    50% {transform: translateY(-20px);}
}


/* ------------------- FOOTER ------------------- */
#custom-footer {
    width: 100%;
    padding: 20px 0;
    margin-top: auto; /* pushes footer to the bottom */
    text-align: center;
    background-color: rgba(0, 0, 0, 0.1);
    border-top: 2px solid rgba(200, 200, 200, 0.45); /* semi-transparent gray */
}

#custom-footer a {
    font-size: 1.2em;
    font-weight: bold;
    text-decoration: none;

    /* gradient text */
    background: linear-gradient(90deg, #ff9a9e, #fad0c4, #fad0c4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

#custom-footer a:hover {
    opacity: 0.7;
}
            
[data-testid="stFileUploader"] > label {
    color:#9CA3AF !important;
}
            
.st-emotion-cache-ysk9xe {
    color: #7A7A7A !important;
}
            
.st-emotion-cache-9ycgxx{
    margin-bottom: 0.25rem;
    color: #000;
    box-sizing: border-box;
    font-size: 18px;
    font-weight: 600;
            }
                       
.stAppHeader {
    padding: 10px;
    height: 70px;
    border-bottom: 1px solid transparent;
    background: linear-gradient(to right, #bc4e9c, #f80759);
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
}
.st-emotion-cache-jdyw56{
            font-size:18px; !important;
            font-weight:600; !important;
            }

.st-emotion-cache-1wbqy5l {
    display: flex;
    gap: 0.5rem;
    font-size: 18px;
    font-weight: 600;
    -webkit-box-align: center;
    align-items: center;
    line-height: 0.9rem;
}
            
.st-emotion-cache-1erivf3 {
    background-color: #fff !important;
    border: 1px solid transparent !important;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
    border-radius: 10px !important
}
.st-emotion-cache-133trn5{
            fill:#ff496d; !important;}
.st-emotion-cache-7oyrr6 {
    color: #9ca3af;
    font-size: 14px;
    line-height: 1.25;
}
            
.st-emotion-cache-clky9d {
    fill: #ff496d;
}

.st-emotion-cache-1n47svx{
            background-color: #ff496d ; !important;
            }

.st-emotion-cache-7oyrr6 {
    color: #ff496d;
    font-size: 14px;
    line-height: 1.25;
}
            
 .st-emotion-cache-7oyrr6 {
    color: #7a7a7a;
    font-size: 14px;
    line-height: 1.25;
}
            
/* Default state */
.st-emotion-cache-1n47svx {
    background-color: #ff496d !important;
    color: white !important;
    border: 2px solid transparent !important;
    transition: 0.25s ease-in-out !important;
    border-radius: 10px !important;
}

/* Hover state */
.st-emotion-cache-1n47svx:hover {
    background-color: transparent !important;   /* slightly darker */
    color: #ff496d !important;
    border: 2px solid #ff496d !important;
    cursor: pointer !important;
}

/* Click (active) state */
.st-emotion-cache-1n47svx:active {
     background-color: transparent !important;   /* slightly darker */
    color: #ff496d !important;
    border: 2px solid #ff496d !important;
    cursor: pointer !important;
    transform: scale(0.97) !important;      /* click effect */
}
            
.st-emotion-cache-ocqkz7 {
    flex-direction: column;
}
            
.st-emotion-cache-13k62yr {
    color-scheme: none;
}
            
.st-emotion-cache-ysk9xe p {
    word-break: break-word;
    margin-bottom: 0px;
    font-size: 16px;
    margin-bottom: 10px;
}

.st-emotion-cache-12xsiil {
    color: #7a7a7a;

/* ----- Button style ----- */
div.stButton > button {
    background-color: #ff496d;
    color: white;
    border-radius: 10px;
    padding: 10px 20px;
    border: none;
    font-size: 1.1rem;
}
div.stButton > button:hover {
    background-color: #fff;
    border: 2px solid #ff496d;
}

.st-emotion-cache-1b2ybts {
    vertical-align: middle;
    overflow: hidden;
    fill: #ff496d;
    display: inline-flex;
    -webkit-box-align: center;
    align-items: center;
    font-size: 1.25rem;
    width: 1.25rem;
    height: 1.25rem;
    flex-shrink: 0;
}

            /* Prediction Section Flexbox */
#predict-wrapper {
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;

    width: 100%;
    padding-top: 50px;
    text-align: center;
}


</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
@keyframes slideInLeft {
    0% {
        opacity: 0;
        transform: translateX(-40px);
    }
    100% {
        opacity: 1;
        transform: translateX(0);
    }
}

.custom-box {
    background: #ffffff;
    padding: 22px 28px;
    border-radius: 18px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.12);
    animation: slideInLeft 1.2s ease;
    border: 1px solid #f0f0f0;
    font-size: 17px;
    color: #6a6a6a;
    line-height: 1.6;
    margin-top: 100px;
            ul{
            li::marker{ 
            color: #ff496d;
            }
            }
}
</style>

<div class="custom-box">
<h3>About Project</h3>
<p>
Music. Experts have been trying for a long time to understand sound and what differenciates one song from another. How to visualize sound. What makes a tone different from another.
</p>

<p>This data hopefully can give the opportunity to do just that.</p>

<h3>About Dataset</h3>
<h4>Content</h4>

<ul>
<li><b>genres original</b> - A collection of 10 genres with 100 audio files each, all having a length of 30 seconds (the famous GTZAN dataset, the MNIST of sounds)</li>

<li><b>List of Genres</b> - blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock</li>

<li><b>images original</b> - A visual representation for each audio file. One way to classify data is through neural networks. Because NNs (like CNN, what we will be using today) usually take in some sort of image representation, the audio files were converted to Mel Spectrograms to make this possible.</li>

<li><b>2 CSV files</b> - Containing features of the audio files. One file has for each song (30 seconds long) a mean and variance computed over multiple features that can be extracted from an audio file. The other file has the same structure, but the songs were split before into 3 seconds audio files (this way increasing 10 times the amount of data we fuel into our classification models). With data, more is always better.</li>
</ul>

</div>
""", unsafe_allow_html=True)

# ------------------ Prediction Section ------------------
st.markdown('<div id="predict-wrapper">', unsafe_allow_html=True)
st.header("Genre Classification")
test_mp3 = st.file_uploader("Upload an audio file", type=["mp3"])

if test_mp3 is not None:
    filepath = 'Test_Music/' + test_mp3.name

col1, col2 = st.columns([1, 1])

with col1:
    if st.button("Play Audio"):
        if test_mp3 is not None:
            st.audio(test_mp3)

with col2:
    if st.button("Predict"):
        if test_mp3 is not None:
            # Custom spinner color
            st.markdown("""
            <style>
            .st-emotion-cache-gfp0l5 {
    font-size: 14px;
    width: 1.375rem;
    height: 1.375rem;
    border-width: 0.2rem;
    padding: 0px;
    margin: 0px;
    border-color: #ff496d rgba(250, 250, 250, 0.2) rgba(250, 250, 250, 0.2);
    -webkit-box-flex: 0;
    flex-grow: 0;
    flex-shrink: 0;
}
            </style>
            """, unsafe_allow_html=True)

            # Show spinner while computing
            with st.spinner("Please Wait.."):
                X_test = load_and_preprocess_data(filepath)
                result_index = model_prediction(X_test)
                st.balloons()

                # Save prediction to session_state
                st.session_state['genre'] = result_index

st.markdown('</div>', unsafe_allow_html=True)


# ---------------- Prediction Display Card (outside button) ----------------
if 'genre' in st.session_state:
    label = ['blues','classical','country','disco','hiphop','jazz','metal','pop','reggae','rock']
    genre = label[st.session_state['genre']]

    st.markdown(f"""
    <div style="
        width: 100%;
        text-align: center;
        padding: 50px 20px;
        font-size: 2.2em;
        font-weight: 700;
        color: #7a7a7a;
        background: #fff;
        border-radius: 20px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.15);
        margin-top: 30px;
        animation: fadeIn 1.5s ease;
    ">
        <span style='color:#7a7a7a;'>Model Prediction:</span> 
        It's a <span style='color:#ff496d;'>{genre}</span> music!
    </div>
    """, unsafe_allow_html=True)



# ------------------ Footer ------------------
st.markdown("""
<!-- Footer -->
<div id="custom-footer">
    Check out my GitHub:  
    <a href="https://github.com/Neginabbas" target="_blank">Negin Abbasi</a>
</div>

<style>
#custom-footer {
    width: 100%;                    /* full width */
    padding: 25px 0;                /* vertical padding */
    margin-top: 60px;               /* space above footer */
    text-align: center;
    background: linear-gradient(to right, #bc4e9c, #f80759);
    color: white;
    font-weight: 600;
    border-top: 2px solid rgba(255,255,255,0.3); /* semi-transparent border */
    box-sizing: border-box;         /* ensures full width */
}

#custom-footer a {
    background: linear-gradient(90deg, #ff9a9e, #fad0c4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 1.1em;
    font-weight: bold;
    text-decoration: none;
}

#custom-footer a:hover {
    opacity: 0.8;
}
</style>
""", unsafe_allow_html=True)




