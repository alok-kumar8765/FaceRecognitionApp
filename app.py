import os
import re
import time
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from deepface import DeepFace

# --- DIRECTORIES & CONFIG ---
APP_DIR = Path(__file__).resolve().parent
DB_DIR = APP_DIR / "face_database"
DB_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "Facenet512"
DETECTOR = "opencv"
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

st.set_page_config(
    page_title="Face Recognition Studio",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM TRENDY CSS STYLING ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stButton>button { width: 100%; border-radius: 8px; font-weight: bold; transition: 0.3s; }
    .stButton>button:hover { border-color: #00ff80; color: #00ff80; }
    .success-box { padding: 1.2rem; border-radius: 10px; background-color: rgba(0, 255, 128, 0.1); border: 1px solid #00ff80; margin-top: 1rem; }
    .error-box { padding: 1.2rem; border-radius: 10px; background-color: rgba(255, 75, 75, 0.1); border: 1px solid #ff4b4b; margin-top: 1rem; }
    </style>
""", unsafe_allow_html=True)

# --- HELPER UTILITIES ---
def image_to_bgr(uploaded_file):
    raw_bytes = uploaded_file.getvalue()
    image_array = np.frombuffer(raw_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Image read nahi ho paayi.")
    return image

def validate_single_face(image):
    try:
        faces = DeepFace.extract_faces(
            img_path=image,
            detector_backend=DETECTOR,
            enforce_detection=True,
            align=True
        )
        if len(faces) != 1:
            raise ValueError(f"Ek hi face hona chahiye. Detected faces: {len(faces)}")
        return True
    except Exception as e:
        # Fallback check with enforce_detection=False if strict detection fails
        try:
            faces = DeepFace.extract_faces(
                img_path=image,
                detector_backend=DETECTOR,
                enforce_detection=False,
                align=True
            )
            if len(faces) == 0:
                raise ValueError("No face detected in the image.")
        except Exception:
            raise ValueError(f"Face validation failed: {str(e)}")

def safe_person_name(name):
    name = name.strip()
    if not re.fullmatch(r"[A-Za-z0-9\-_]{0,39}", name):
        raise ValueError("Naam mein letters, numbers, spaces, _ ya - use karo.")
    return name

def get_registered_images():
    return [
        path for path in DB_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in ALLOWED_EXTENSIONS
    ]

# --- APP HEADER ---
st.title("👤 Face Recognition Studio")
st.caption("Python • Streamlit • OpenCV • DeepFace | Advanced AI Identification Platform")

# --- NAVIGATION TABS ---
tab1, tab2, tab3 = st.tabs(["📝 1. Register Face", "🔍 2. Recognize Face", "📁 3. Local Database"])

# --- TAB 1: REGISTER FACE ---
with tab1:
    st.header("Register an Authorized Person")
    st.warning("⚠️ Educational demo only. Use face images only with the person's informed permission.")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        person_name = st.text_input("Person's name")
        uploaded_reg_image = st.file_uploader("Upload one clear face photo", type=["jpg", "jpeg", "png"])
        permission = st.checkbox("I have this person's permission to register and process their face image.")
        
        register_btn = st.button("Register Face", type="primary")

    if register_btn:
        if not person_name:
            st.error("Please enter a person's name.")
        elif not uploaded_reg_image:
            st.error("Please upload a face photo.")
        elif not permission:
            st.error("You must confirm permission to proceed.")
        else:
            try:
                with st.spinner("Processing and validating face..."):
                    clean_name = safe_person_name(person_name)
                    bgr_image = image_to_bgr(uploaded_reg_image)
                    validate_single_face(bgr_image)
                    
                    person_folder = DB_DIR / clean_name
                    person_folder.mkdir(parents=True, exist_ok=True)
                    
                    file_path = person_folder / f"face_{int(time.time())}.jpg"
                    cv2.imwrite(str(file_path), bgr_image)
                    
                with col2:
                    st.markdown(f"""
                        <div class="success-box">
                            <h3>✅ {clean_name} successfully registered!</h3>
                        </div>
                    """, unsafe_allow_html=True)
                    st.image(bgr_image[:, :, ::-1], width=250, caption="Registered Face")
            except Exception as e:
                st.error(f"❌ Error: {e}")

# --- TAB 2: RECOGNIZE FACE ---
with tab2:
    st.header("Recognize a Registered Face")
    st.warning("⚠️ Educational demo only. Results can be wrong; do not use this app for security decisions.")
    
    rec_permission = st.checkbox("I have permission to process the person shown in this image.", key="rec_perm")
    source_choice = st.radio("Image source", ["Webcam", "Upload image"])
    
    target_image = None
    if source_choice == "Webcam":
        cam_img = st.camera_input("Capture Image")
        if cam_img is not None:
            target_image = image_to_bgr(cam_img)
    else:
        up_img = st.file_uploader("Upload image to check", type=["jpg", "jpeg", "png"], key="check_img")
        if up_img is not None:
            target_image = image_to_bgr(up_img)
            
    if target_image is not None and rec_permission:
        st.image(target_image[:, :, ::-1], width=350, caption="Image being checked")
        
        if st.button("Verify Match", type="primary"):
            registered_files = get_registered_images()
            if not registered_files:
                st.error("Database is empty! Please register faces first in the 'Register Face' tab.")
            else:
                with st.spinner("Analyzing face and searching database..."):
                    temp_path = APP_DIR / "temp_target.jpg"
                    try:
                        cv2.imwrite(str(temp_path), target_image)
                        
                        # Attempt recognition with strict enforcement first, fallback to False if needed
                        try:
                            dfs = DeepFace.find(
                                img_path=str(temp_path),
                                db_path=str(DB_DIR),
                                model_name=MODEL_NAME,
                                detector_backend=DETECTOR,
                                enforce_detection=True,
                                #distance_metric="cosine",
                                #silent=True
                            )
                        except Exception:
                            dfs = DeepFace.find(
                                img_path=str(temp_path),
                                db_path=str(DB_DIR),
                                model_name=MODEL_NAME,
                                detector_backend=DETECTOR,
                                enforce_detection=False,
                                #distance_metric="cosine",
                                #silent=True
                            )
                        
                        if temp_path.exists():
                            temp_path.unlink()
                            
                        if len(dfs) > 0 and not dfs[0].empty:
                            result_df = dfs[0]
                            best_match = result_df.iloc[0]
                            matched_path = Path(best_match['identity'])
                            matched_name = matched_path.parent.name
                            
                            # Dynamically find the distance column (e.g., Facenet512_cosine)
                            dist_col = [col for col in result_df.columns if 'cosine' in col or 'distance' in col]
                            distance = best_match[dist_col[0]] if dist_col else 0.0
                            
                            THRESHOLD = 0.40
                            if distance <= THRESHOLD:
                                st.markdown(f"""
                                    <div class="success-box">
                                        <h3>✅ MATCH FOUND: {matched_name}</h3>
                                        <p>Distance score: <b>{distance:.4f}</b></p>
                                    </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown(f"""
                                    <div class="error-box">
                                        <h3>❌ UNKNOWN</h3>
                                        <p>Closest match was {matched_name} with distance <b>{distance:.4f}</b>.</p>
                                    </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.markdown("""
                                <div class="error-box">
                                    <h3>❌ UNKNOWN</h3>
                                    <p>No accepted match found in database.</p>
                                </div>
                            """, unsafe_allow_html=True)
                    except Exception as e:
                        if temp_path.exists():
                            temp_path.unlink()
                        st.error(f"Recognition error: {e}")
    elif target_image is not None and not rec_permission:
        st.warning("Please check the permission box to proceed with recognition.")

# --- TAB 3: LOCAL DATABASE EXPLORER ---
with tab3:
    st.header("📁 Local Database Explorer")
    registered_files = get_registered_images()
    
    if not registered_files:
        st.info("No faces registered in database yet.")
    else:
        st.write(f"Total registered face files: **{len(registered_files)}**")
        
        persons = {}
        for file_path in registered_files:
            p_name = file_path.parent.name
            if p_name not in persons:
                persons[p_name] = []
            persons[p_name].append(file_path)
            
        for p_name, paths in persons.items():
            with st.expander(f"👤 {p_name} ({len(paths)} photos)"):
                cols = st.columns(4)
                for idx, p in enumerate(paths):
                    with cols[idx % 4]:
                        st.image(str(p), use_container_width=True, caption=p.name)