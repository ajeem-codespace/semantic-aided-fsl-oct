import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
from sentence_transformers import SentenceTransformer
import numpy as np
import io 
import os 


IMG_HEIGHT, IMG_WIDTH = 224, 224
VISION_ENCODER_NAME = 'resnet50' 
VISUAL_FEATURE_DIM = 2048
TEXT_ENCODER_NAME = 'all-MiniLM-L6-v2'
TEXT_FEATURE_DIM = 384
SEMALIGN_HIDDEN_DIM = 1024 

FUSION_FACTOR_K = 0.5 

# Define device 
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    DEVICE = torch.device("mps")
    print("Using Apple Metal (MPS) GPU for PyTorch.")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f"Using NVIDIA CUDA GPU: {torch.cuda.get_device_name(0)} for PyTorch.")
else:
    DEVICE = torch.device("cpu")
    print("Using CPU for PyTorch.")

st.set_page_config(layout="wide", page_title="Few-Shot OCT Classifier")

#  Model Definitions (used to make the model - can be found in notebook)
class SemAlignNet(nn.Module):
    def __init__(self, visual_dim, text_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(visual_dim + text_dim, hidden_dim)
        self.relu = nn.LeakyReLU(0.1)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, visual_features, semantic_features):
        combined_features = torch.cat((visual_features, semantic_features), dim=1)
        hidden_representation = self.relu(self.fc1(combined_features))
        output_prototype = self.fc2(hidden_representation)
        return output_prototype

# Load Models and Encoders
@st.cache_resource 
def load_resources(device_to_use):
    # Load Vision Encoder
    vision_enc = models.resnet50(weights=None) # Load architecture
    vision_enc.fc = nn.Identity() # To output features
    
    vision_encoder_path = "fine_tuned_oct_vision_encoder.pth"
    if not os.path.exists(vision_encoder_path):
        st.error(f"Error: Vision encoder weights ('{vision_encoder_path}') not found. Place it in the app directory.")
        return None, None, None
    try:
        vision_enc.load_state_dict(torch.load(vision_encoder_path, map_location=device_to_use))
    except Exception as e:
        st.error(f"Error loading vision encoder state_dict: {e}")
        return None, None, None
    vision_enc.to(device_to_use)
    vision_enc.eval()

    # Load Text Encoder
    try:
        text_enc = SentenceTransformer(TEXT_ENCODER_NAME, device=str(device_to_use))
    except Exception as e:
        st.error(f"Error loading text encoder '{TEXT_ENCODER_NAME}': {e}")
        return None, None, None

    # Load SemAlignNet
    semalign_m = SemAlignNet(VISUAL_FEATURE_DIM, TEXT_FEATURE_DIM, SEMALIGN_HIDDEN_DIM, VISUAL_FEATURE_DIM)
    semalign_net_path = "oct_semalign_net.pth"
    if not os.path.exists(semalign_net_path):
        st.error(f"Error: SemAlignNet weights ('{semalign_net_path}') not found. Place it in the app directory.")
        return None, None, None
    try:
        semalign_m.load_state_dict(torch.load(semalign_net_path, map_location=device_to_use))
    except Exception as e:
        st.error(f"Error loading SemAlignNet state_dict: {e}")
        return None, None, None
    semalign_m.to(device_to_use)
    semalign_m.eval()
    
    st.success("Models loaded successfully!")
    return vision_enc, text_enc, semalign_m


if 'models_loaded' not in st.session_state or not st.session_state.models_loaded:
    vision_encoder_f, text_encoder, semalign_net = load_resources(DEVICE)
    if vision_encoder_f and text_encoder and semalign_net:
        st.session_state.models_loaded = True
        st.session_state.vision_encoder_f = vision_encoder_f
        st.session_state.text_encoder = text_encoder
        st.session_state.semalign_net = semalign_net
    else:
        st.error("Failed to load one or more models. The app may not function correctly.")
        
        st.stop() 
elif 'vision_encoder_f' not in st.session_state : 
    vision_encoder_f, text_encoder, semalign_net = load_resources(DEVICE)
    if vision_encoder_f and text_encoder and semalign_net:
        st.session_state.models_loaded = True
        st.session_state.vision_encoder_f = vision_encoder_f
        st.session_state.text_encoder = text_encoder
        st.session_state.semalign_net = semalign_net


# Define image transforms (inference version - no augmentation)
transform_oct_infer = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.Grayscale(num_output_channels=3), # Ensure 3 channels for ResNet
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

#Helper function to generate prototype (from FSL evaluation logic)
def generate_prototype_for_new_class(support_images_tensor, semantic_description_str,
                                     vision_enc, text_enc, semalign_m, k_fusion, device_to_use):
    if support_images_tensor is None or not semantic_description_str or vision_enc is None or text_enc is None or semalign_m is None:
        st.error("Error in generate_prototype: One or more model components are None or inputs are missing.")
        return None
        
    num_support_shots = support_images_tensor.shape[0]

    with torch.no_grad():
        visual_features_support = vision_enc(support_images_tensor.to(device_to_use)) 
        u_t_new = torch.mean(visual_features_support, dim=0) 

        semantic_features_new_batch = text_enc.encode([semantic_description_str], convert_to_tensor=True, device=str(device_to_use))
        semantic_features_new = semantic_features_new_batch[0].float() 
        
        semantic_features_new_expanded = semantic_features_new.unsqueeze(0).expand(num_support_shots, -1)
        
        reconstructed_prototypes_new = semalign_m(visual_features_support, semantic_features_new_expanded)
        r_t_new = torch.mean(reconstructed_prototypes_new, dim=0)

        p_t_new = k_fusion * r_t_new + (1 - k_fusion) * u_t_new
    return p_t_new

# --- Streamlit UI ---
st.title("Few-Shot OCT Disease Learner & Classifier")
st.markdown("""
This application demonstrates the ability to learn a **new retinal condition** from just a few OCT images and a text description, 
and then classify a query image against this newly learned condition. This is based on the "Simple Semantic-Aided Few-Shot Learning" methodology.
""")

# Initialize session state
if 'new_disease_prototype' not in st.session_state:
    st.session_state.new_disease_prototype = None
if 'new_disease_name' not in st.session_state:
    st.session_state.new_disease_name = ""
if 'new_disease_semantic_description' not in st.session_state: 
    st.session_state.new_disease_semantic_description = ""
if 'normal_prototype' not in st.session_state: 
    st.session_state.normal_prototype = None
    st.session_state.normal_semantic_description = "Normal OCT: A normal retina on OCT shows distinct, continuous retinal layers with a preserved foveal contour and foveal pit. There is an absence of intraretinal or subretinal fluid, no hemorrhages, drusen, or other visible pathologies. Retinal thickness parameters are within normal ranges for the patient's demographic" 

# --- Sidebar for "Teaching" a New Disease ---
with st.sidebar:
    st.header("First Step: Teach a New Disease")
    new_disease_name_input = st.text_input("Name of the new disease:", 
                                           value=st.session_state.get("new_disease_name_teaching_buffer", "My New OCT Condition"), 
                                           key="disease_name_teach_input")
    
    K_DEMO_SHOT = st.number_input("Number of example images (K-shots):", min_value=1, max_value=10, value=3, key="k_shot_teach")
    
    support_files = st.file_uploader(f"Upload {K_DEMO_SHOT} support OCT images for '{new_disease_name_input}'",
                                     type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="support_uploader")
    
    semantic_description_input_area = st.text_area(f"Provide a detailed OCT visual description for '{new_disease_name_input}':", 
                                           height=250,
                                           placeholder="e.g., OCT shows significant subretinal fluid, RPE undulation, and photoreceptor layer disruption...",
                                           key="semantic_desc_teach_area")

    if st.button(" Learn Disease & Generate Prototype", type="primary", key="learn_button"):
        if not st.session_state.get('models_loaded', False): 
            st.error("Models are not loaded. Please ensure model files are present and refresh the page.")
        elif support_files and len(support_files) == K_DEMO_SHOT and semantic_description_input_area and new_disease_name_input:
            st.session_state.new_disease_name_teaching_buffer = new_disease_name_input 
            with st.spinner(f"Learning '{new_disease_name_input}' from {K_DEMO_SHOT} images..."):
                try:
                    support_images_pil = [Image.open(io.BytesIO(file.getvalue())).convert('L') for file in support_files]
                    support_images_tensor = torch.stack([transform_oct_infer(img) for img in support_images_pil])
                    
                    p_t_new = generate_prototype_for_new_class(
                        support_images_tensor, 
                        semantic_description_input_area,
                        st.session_state.vision_encoder_f, 
                        st.session_state.text_encoder, 
                        st.session_state.semalign_net, 
                        FUSION_FACTOR_K, 
                        DEVICE
                    )
                    
                    if p_t_new is not None:
                        st.session_state.new_disease_prototype = p_t_new
                        st.session_state.new_disease_name = new_disease_name_input
                        st.session_state.new_disease_semantic_description = semantic_description_input_area
                        st.success(f"Prototype generated and stored for '{new_disease_name_input}'!")
                        st.balloons()
                    else:
                        st.error("Failed to generate prototype. Check inputs or model loading status.")
                except Exception as e:
                    st.error(f"An error occurred during prototype generation: {e}")
                    st.exception(e) 
        else:
            st.warning(f"Please upload exactly {K_DEMO_SHOT} images, provide a disease name, and a semantic description.")

    # Optional: Generate a "NORMAL" prototype for comparison
    normal_sample_image_dir = os.path.join("assets", "normal_oct_samples") # Path relative to app.py
    if st.button("Load 'NORMAL' Reference Prototype", key="load_normal_button"):
        if not st.session_state.get('models_loaded', False):
             st.error("Models are not loaded. Please ensure model files are present and refresh the page.")
        else:
            normal_sample_paths = []
            if os.path.exists(normal_sample_image_dir) and os.path.isdir(normal_sample_image_dir):
                normal_sample_paths = [os.path.join(normal_sample_image_dir, f) for f in os.listdir(normal_sample_image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:K_DEMO_SHOT]
            
            if len(normal_sample_paths) == K_DEMO_SHOT:
                with st.spinner("Generating 'NORMAL' prototype..."):
                    try:
                        normal_images_pil = [Image.open(p).convert('L') for p in normal_sample_paths]
                        normal_images_tensor = torch.stack([transform_oct_infer(img) for img in normal_images_pil])
                        
                        p_t_normal = generate_prototype_for_new_class(
                            normal_images_tensor,
                            st.session_state.normal_semantic_description,
                            st.session_state.vision_encoder_f,
                            st.session_state.text_encoder,
                            st.session_state.semalign_net,
                            FUSION_FACTOR_K,
                            DEVICE
                        )
                        if p_t_normal is not None:
                            st.session_state.normal_prototype = p_t_normal
                            st.success("'NORMAL' reference prototype generated and stored.")
                        else:
                            st.error("Failed to generate 'NORMAL' prototype.")
                    except Exception as e:
                        st.error(f"Error generating 'NORMAL' prototype: {e}")
                        st.exception(e)
            else:
                st.warning(f"Could not find {K_DEMO_SHOT} sample images for 'NORMAL' in '{normal_sample_image_dir}'. Please create this folder and add images, or implement user upload for NORMAL samples.")

# --- Main Area for Querying ---
st.header("Second Step: Classify a Query OCT Image")

if st.session_state.new_disease_prototype is None and st.session_state.normal_prototype is None:
    st.info("⬅️ Please teach a new disease or load the 'NORMAL' reference prototype using the sidebar first.")
else:
    query_file = st.file_uploader("Upload a query OCT image:", type=["jpg", "jpeg", "png"], key="query_uploader")

    if query_file:
        query_image_pil = Image.open(io.BytesIO(query_file.getvalue())).convert('L')
        
        col1, col2 = st.columns([1, 2]) 
        with col1:
            st.image(query_image_pil, caption="Uploaded Query Image", use_container_width=True)

        if st.button("🔍 Classify Query Image", type="primary", key="classify_button"):
            if not st.session_state.get('models_loaded', False) or st.session_state.vision_encoder_f is None:
                st.error("Models not loaded. Please ensure model files are present and refresh the page.")
            else:
                with st.spinner("Classifying..."):
                    query_image_tensor = transform_oct_infer(query_image_pil).unsqueeze(0).to(DEVICE)
                    
                    with torch.no_grad():
                        query_features = st.session_state.vision_encoder_f(query_image_tensor).squeeze(0) 
                        query_features_norm = torch.nn.functional.normalize(query_features, p=2, dim=0)

                        results = []
                        # Compare with the "newly taught" disease prototype
                        if st.session_state.new_disease_prototype is not None:
                            new_disease_proto_norm = torch.nn.functional.normalize(st.session_state.new_disease_prototype, p=2, dim=0)
                            similarity_new = torch.dot(query_features_norm, new_disease_proto_norm).item()
                            results.append({"name": st.session_state.new_disease_name, 
                                            "score": similarity_new, 
                                            "description": st.session_state.new_disease_semantic_description})
                        
                        # Compare with the "NORMAL" reference prototype (if loaded)
                        if st.session_state.normal_prototype is not None:
                            normal_proto_norm = torch.nn.functional.normalize(st.session_state.normal_prototype, p=2, dim=0)
                            similarity_normal = torch.dot(query_features_norm, normal_proto_norm).item()
                            results.append({"name": "NORMAL (Reference)", 
                                            "score": similarity_normal, 
                                            "description": st.session_state.normal_semantic_description})
                        
                        if results:
                            results.sort(key=lambda x: x["score"], reverse=True) 
                            predicted_class_name = results[0]["name"]
                            confidence_score = results[0]["score"]
                            predicted_description = results[0]["description"]

                            with col2:
                                st.subheader("Classification Result:")
                                st.success(f"Predicted as: **{predicted_class_name}**")
                                st.metric(label=f"Similarity Score to {predicted_class_name}", value=f"{confidence_score:.3f}")

                                if len(results) > 1:
                                    st.write("Other similarities:")
                                    for res in results[1:]:
                                        st.write(f"- {res['name']}: {res['score']:.3f}")
                                
                                if predicted_description:
                                     with st.expander(f"View Semantic Description for {predicted_class_name}", expanded=True):
                                        st.info(predicted_description)
                        else:
                            with col2:
                                st.error("No prototypes available to classify against. Please teach a new disease or load a reference prototype.")
        st.markdown("---")

st.sidebar.header("About")
st.sidebar.info(
    """
    This application demonstrates a capstone project implementing 
    the 'Simple Semantic-Aided Few-Shot Learning' paper 
    (Zhang et al., arXiv:2311.18649) for OCT image classification.
    It allows users to "teach" the system a new retinal condition 
    using a few example images and a text description.
    """
)


