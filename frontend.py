import streamlit as st
import requests
from PIL import Image
import io

# 1. Page Config
st.set_page_config(page_title="Face Matcher AI", layout="wide")

st.title("🔍 Face Matcher AI")
st.markdown("Upload a photo to find the most similar people in our **Qdrant** database.")

# 2. Sidebar for User Input
with st.sidebar:
    st.header("Upload Image")
    uploaded_file = st.file_uploader("Choose a face...", type=["jpg", "png", "jpeg", "webp"])
    
    # Optional: You can add the Gender Filter here later if you update the backend
    # gender_filter = st.selectbox("Filter by Gender", ["All", "Male", "Female"])

# 3. Main Logic
if uploaded_file is not None:
    # Display the uploaded image
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Your Upload")
        st.image(uploaded_file, use_container_width=True)
        
    with col2:
        st.subheader("Search Results")
        
        # Search Button
        if st.button("Find Matches", type="primary"):
            with st.spinner("Searching vector database..."):
                try:
                    # Prepare the file for the API request
                    files = {
                        "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
                    }
                
                    response = requests.post("http://127.0.0.1:8001/search", files=files)
                    
                    if response.status_code == 200:
                        data = response.json()
                        matches = data.get("matches", [])
                        
                        if not matches:
                            st.warning("No matches found.")
                        else:
                            # Display results in 3 columns
                            cols = st.columns(3)
                            for idx, match in enumerate(matches):
                                with cols[idx]:
                                    # Show Image from the URL provided by FastAPI
                                    st.image(match["image_url"], use_container_width=True)
                                    
                                    # Show Details
                                    st.success(f"**{match['similarity_score']*100:.1f}% Match**")
                                    st.caption(f"Gender: {match['gender']}")
                                    st.caption(f"File: {match['filename']}")
                    else:
                        st.error(f"Error {response.status_code}: {response.text}")
                        
                except requests.exceptions.ConnectionError:
                    st.error("❌ Could not connect to backend. Is FastAPI running?")
                except Exception as e:
                    st.error(f"An error occurred: {e}")

else:
    # Instructions when no file is uploaded
    st.info("👈 Please upload an image in the sidebar to get started.")