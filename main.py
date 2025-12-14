import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import joblib
import json
import time
import os

# --- KONFIGURACJA STRONY ---
st.set_page_config(
    page_title="Tłumacz Migowy AI", 
    page_icon="👋", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- UKRYWANIE ELEMENTÓW INTERFEJSU (CSS) ---
st.markdown("""
<style>
    /* Ukrywa menu (trzy kropki) w prawym górnym rogu */
    #MainMenu {visibility: hidden;}
    
    /* Ukrywa stopkę "Made with Streamlit" */
    footer {visibility: hidden;}
    
    /* Ukrywa nagłówek (ten kolorowy pasek na samej górze) */
    header {visibility: hidden;}
    
    /* Ukrywa animację "Running..." w prawym górnym rogu ekranu */
    .stStatusWidget {visibility: hidden;}

    /* Reszta stylów dla czatu */
    .stApp {
        background-color: #0e1117;
    }
    .chat-msg {
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
        background-color: #262730;
        color: white;
        border-left: 5px solid #00ff00;
    }
</style>
""", unsafe_allow_html=True)

# --- ŁADOWANIE ZASOBÓW (CACHE) ---
# Używamy cache, żeby nie ładować modelu przy każdym odświeżeniu strony
@st.cache_resource
def load_model_and_labels():
    try:
        clf = joblib.load("model.joblib")
        with open("labels.json", "r") as f:
            labels = json.load(f)["labels"]
        return clf, labels
    except Exception as e:
        st.error(f"Błąd ładowania modelu: {e}")
        return None, None

@st.cache_resource
def load_mediapipe():
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    return mp_hands, mp_draw, hands

# --- FUNKCJE POMOCNICZE ---
def normalize_landmarks(flat_xyz):
    pts = np.array(flat_xyz, dtype=np.float32).reshape(-1, 3)
    pts -= pts[0]
    scale = np.linalg.norm(pts, axis=1).max()
    if scale > 1e-6:
        pts /= scale
    return pts.reshape(-1)

# --- INICJALIZACJA STANU (SESSION STATE) ---
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []
if 'current_sentence' not in st.session_state:
    st.session_state['current_sentence'] = ""

# --- GŁÓWNA APLIKACJA ---
def main():
    # 1. Pasek boczny
    st.sidebar.title("⚙️ Ustawienia")
    camera_index = st.sidebar.selectbox("Wybierz kamerę", [0, 1, 2], index=0)
    confidence_threshold = st.sidebar.slider("Czułość modelu", 0.0, 1.0, 0.6)
    hold_duration = st.sidebar.slider("Czas przytrzymania (s)", 0.5, 3.0, 1.5)
    
    use_webcam = st.sidebar.toggle("Uruchom Kamerę", value=False)
    
    # Przycisk czyszczenia historii
    if st.sidebar.button("Wyczyść historię"):
        st.session_state['chat_history'] = []
        st.session_state['current_sentence'] = ""
        st.rerun()

    # 2. Główny układ (Dwie kolumny)
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("🎥 Podgląd na żywo")
        frame_placeholder = st.empty()  # Miejsce na wideo
        status_text = st.empty()        # Miejsce na status

    with col2:
        st.subheader("💬 Twoja rozmowa")
        # Wyświetlanie historii
        chat_container = st.container()
        with chat_container:
            for msg in reversed(st.session_state['chat_history']):
                st.markdown(f"<div class='chat-msg'>{msg}</div>", unsafe_allow_html=True)
            
            if not st.session_state['chat_history']:
                st.info("Tutaj pojawią się wysłane zdania.")

    # 3. Logika pętli wideo
    if use_webcam:
        clf, labels = load_model_and_labels()
        mp_hands, mp_draw, hands = load_mediapipe()

        if clf is None:
            st.stop()

        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Zmienne lokalne dla pętli
        last_gesture = None
        gesture_start_time = 0
        last_action_time = 0
        
        # Pętla while wewnątrz Streamlit
        while cap.isOpened() and use_webcam:
            ret, frame = cap.read()
            if not ret:
                st.error("Nie można odczytać klatki z kamery.")
                break

            # Przetwarzanie obrazu
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb_frame)
            
            current_prediction = None
            conf_score = 0.0

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    landmarks = []
                    for lm in hand_landmarks.landmark:
                        landmarks.extend([lm.x, lm.y, lm.z])
                    
                    features = normalize_landmarks(landmarks).reshape(1, -1)
                    
                    try:
                        probs = clf.predict_proba(features)[0]
                        idx = np.argmax(probs)
                        conf_score = probs[idx]
                        
                        if conf_score > confidence_threshold:
                            current_prediction = labels[idx]
                    except:
                        pass

            # Logika czasu i akcji
            now = time.time()
            progress = 0.0
            
            if current_prediction and current_prediction == last_gesture:
                duration = now - gesture_start_time
                progress = min(duration / hold_duration, 1.0)
                
                if duration >= hold_duration and (now - last_action_time) > 1.0:
                    # Wykonanie akcji
                    if current_prediction == "-":
                        st.session_state['current_sentence'] = st.session_state['current_sentence'][:-1]
                    
                    elif current_prediction == ".":
                        if st.session_state['current_sentence']:
                            # Dodaj do historii
                            timestamp = time.strftime("%H:%M:%S")
                            msg = f"[{timestamp}] {st.session_state['current_sentence']}"
                            st.session_state['chat_history'].append(msg)
                            st.session_state['current_sentence'] = ""
                            st.rerun() # Odśwież stronę żeby zaktualizować czat po prawej
                    
                    elif current_prediction == "space":
                        st.session_state['current_sentence'] += " "
                    
                    else:
                        st.session_state['current_sentence'] += current_prediction
                    
                    last_action_time = now
                    gesture_start_time = now
            else:
                last_gesture = current_prediction
                gesture_start_time = now
                progress = 0.0

            # RYSOWANIE GUI NA KLATCE WIDEO (dla płynności)
            # Pasek postępu
            bar_width = int(200 * progress)
            color = (0, 255, 255) if progress < 1.0 else (0, 255, 0)
            cv2.rectangle(frame, (220, 10), (220 + bar_width, 30), color, -1)
            cv2.rectangle(frame, (220, 10), (420, 30), (200, 200, 200), 2)

            # Aktualnie wykryty znak
            if current_prediction:
                cv2.putText(frame, f"{current_prediction} ({conf_score:.0%})", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Aktualnie budowane zdanie (na dole klatki)
            cv2.rectangle(frame, (0, 420), (640, 480), (0, 0, 0), -1)
            cv2.putText(frame, f"TEXT: {st.session_state['current_sentence']}", (10, 460), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # Konwersja z powrotem na RGB dla Streamlit
            # Streamlit oczekuje RGB, OpenCV ma BGR (ale już zmieniliśmy na początku pętli na RGB do MediaPipe)
            # Więc musimy tylko narysować na `frame` (który jest BGR bo cv2 funkcje rysowania działają na nim)
            # A potem przekonwertować wynikowy obraz z interfejsem na RGB
            frame_rgb_display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Aktualizacja obrazu w Streamlit
            frame_placeholder.image(frame_rgb_display, channels="RGB")
            
            # Małe opóźnienie dla stabilności
            time.sleep(0.01)

        cap.release()
    else:
        st.info("Włącz przełącznik 'Uruchom Kamerę' w menu bocznym, aby rozpocząć.")

if __name__ == "__main__":
    main()