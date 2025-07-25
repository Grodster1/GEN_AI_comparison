import streamlit as st
import torch
import torch.functional as F
from data_prep import get_cifar10
from vae_model import VAE
from cvae_model import ConditionalVAE
from cgan_model import CGAN
from visualization import interpolate_latent_space, visualize_latent_space
from utils import load_checkpoint
import os
import numpy as np
import torchvision.utils as vutils

CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer', 
    'dog', 'frog', 'horse', 'ship', 'truck'
]


#Without caching when user interacts with app the entire Python script has to be rerun
@st.cache_resource
def load_model_cached(model_type, checkpoint, device):
    if model_type == "VAE":
        model = VAE(latent_dim=128).to(device)
    elif model_type == "CVAE":
        model = ConditionalVAE(latent_dim=128).to(device)
    elif model_type == "CGAN":
        model = CGAN(latent_dim=100).to(device)

    else:
        return None

    if os.path.exists(checkpoint):
        checkpoint = torch.load(checkpoint, map_location=device)

        if model_type == "VAE":
            model.load_state_dict(checkpoint["model_state_dict"])

        elif model_type == "CVAE":
            model.load_state_dict(checkpoint["model_state_dict"])

        else:
            model.load_state_dict(checkpoint)

        st.success(f"Successfully loaded model {model_type}")
    
    else:
        st.error(f"Checkpoint {checkpoint} does not exist")
        return None
    
    model.eval()
    return model



def generate_images_by_classes(model, model_type, target_class, n_samples, device = "cuda"):
    model.eval()
    with torch.no_grad():
        if model_type == "CVAE":
            z = torch.randn(n_samples, model.latent_dim).to(device)
            y = torch.tensor([target_class] * n_samples, device=device)
            y_one_hot = F.one_hot(y, num_classes=model.num_classes).float()
            samples = model.decode(z, y_one_hot)
        elif model_type == "CGAN":
            noise = model.generate_noise(n_samples)
            labels = torch.full((n_samples,), target_class, device=device)
            samples = model.generator(noise, labels)
        else:
            z = torch.randn(n_samples, model.latent_dim, device=device)
            samples = model.decode(z)
    
    return samples

def images_to_grid(images, nrow=4):
    grid = vutils.make_grid(images.cpu(), nrow=nrow, normalize=True)
    grid = grid.permute(1, 2, 0).numpy()
    return np.clip(grid, 0, 1)


def main():
    # Page config
    st.set_page_config(
        page_title="Generative Models Explorer",
        page_icon="🎨",
        layout="wide"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4, #45B7D1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 3rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">🎨 Generative Models Explorer</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Explore VAE, CGAN, SAGAN, and Diffusion models on CIFAR-10</p>', unsafe_allow_html=True)

    # Device info
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_icon = "🚀" if device.type == "cuda" else "💻"
    st.info(f"{device_icon} Using device: {device}")

    # Load test data
    @st.cache_data
    def load_test_data():
        _, test_loader, _ = get_cifar10(batch_size=64)
        return test_loader
    
    test_loader = load_test_data()

    # Sidebar for model selection
    st.sidebar.header("🔧 Model Configuration")
    
    # Available models (expand this as you add more)
    available_models = {
        "VAE": "checkpoints/vae_final.pth",
        "CVAE": "checkpoints/cvae_final.pth",
        "CGAN": "checkpoints/cgan_epoch_50.pth", 
        "SAGAN": "checkpoints/sagan_epoch_50.pth",
        # "Diffusion": "checkpoints/diffusion_final.pth"  # For future
    }
    
    model_choice = st.sidebar.selectbox(
        "Select Model",
        options=list(available_models.keys()),
        help="Choose which generative model to explore"
    )
    
    # Load selected model
    checkpoint_path = available_models[model_choice]
    model = load_model_cached(model_choice, checkpoint_path, device)
    
    if model is None:
        st.error("❌ Failed to load model. Please check the checkpoint path.")
        return
    
    # Main interface
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("🎛️ Controls")
        
        # Visualization type selection
        viz_options = ["Generate by Class", "Latent Space Interpolation"]
        if model_choice in ["VAE", "CVAE"]:
            viz_options.append("Latent Space Visualization")
        
        viz_type = st.radio("Choose Visualization", viz_options)
        
        # Class-based generation
        if viz_type == "Generate by Class":
            st.subheader("🎯 Generate Images by Class")
            
            if model_choice == "VAE":
                st.info("ℹ️ VAE generates random images (class selection not applicable)")
                target_class = 0  # Dummy value
            else:
                target_class = st.selectbox(
                    "Select CIFAR-10 Class",
                    options=range(10),
                    format_func=lambda x: f"{x}: {CIFAR10_CLASSES[x]}"
                )
            
            num_images = st.slider("Number of Images", 1, 16, 8)
            
            if st.button("🎨 Generate Images", type="primary"):
                with st.spinner("Generating images..."):
                    generated_images = generate_images_by_classes(
                        model, model_choice, target_class, num_images, device
                    )
                    
                    # Display in main area
                    with col2:
                        if model_choice == "VAE":
                            st.header("📸 Generated Random Images")
                        else:
                            st.header(f"📸 Generated {CIFAR10_CLASSES[target_class]} Images")
                        
                        # Create image grid
                        nrow = min(4, num_images)
                        img_grid = images_to_grid(generated_images, nrow=nrow)
                        
                        st.image(img_grid, use_container_width=True)
                        
                        # Show individual images in expandable section
                        with st.expander("🔍 View Individual Images"):
                            cols = st.columns(min(4, num_images))
                            for i in range(num_images):
                                with cols[i % 4]:
                                    single_img = images_to_grid(generated_images[i:i+1], nrow=1)
                                    st.image(single_img, caption=f"Image {i+1}")
        
        # Latent space interpolation
        elif viz_type == "Latent Space Interpolation":
            st.subheader("🔄 Latent Space Interpolation")
            
            if model_choice in ["VAE", "CVAE"]:
                n_steps = st.slider("Interpolation Steps", 5, 20, 10)
                
                if st.button("🔄 Generate Interpolation", type="primary"):
                    with st.spinner("Generating interpolation..."):
                        if model_choice == "CVAE":
                            # For CVAE, we can interpolate with class conditioning
                            st.info("🚧 CVAE interpolation implementation coming soon!")
                            # TODO: Implement CVAE interpolation
                        else:
                            img = interpolate_latent_space(
                                model, test_loader, n_steps=n_steps, 
                                device=device, streamlit=True
                            )
                        
                        with col2:
                            st.header("🔄 Latent Space Interpolation")
                            if model_choice == "VAE":
                                st.image(img, use_container_width=True)
                                st.caption("Smooth transition between two points in latent space")
            else:
                st.info("🚧 Interpolation for GANs coming soon!")
        
        # Latent space visualization (VAE and CVAE)
        elif viz_type == "Latent Space Visualization" and model_choice in ["VAE", "CVAE"]:
            st.subheader("🌌 Latent Space Visualization")
            
            method = st.selectbox("Reduction Method", ["t-SNE", "PCA"])
            n_samples = st.slider("Number of Samples", 100, 2000, 1000)
            
            if st.button("🌌 Generate Visualization", type="primary"):
                with st.spinner("Computing dimensionality reduction..."):
                    fig = visualize_latent_space(
                        model, test_loader, method=method.lower(),
                        n_samples=n_samples, device=device, streamlit=True
                    )
                    
                    with col2:
                        st.header(f"🌌 {method} Latent Space Visualization")
                        st.pyplot(fig)
                        st.caption("Each point represents an encoded image, colored by class")
    
    with col2:
        if 'generated_images' not in locals():
            # Welcome message when no generation has been done yet
            st.header("👋 Welcome!")
            st.info("""
            🎯 **Get Started:**
            1. Choose a model from the sidebar
            2. Select a visualization type
            3. Configure your parameters
            4. Click generate to see results!
            
            🤖 **Available Models:**
            - **VAE**: Variational Autoencoder for smooth latent space
            - **CVAE**: Conditional VAE for class-controlled generation
            - **CGAN**: Conditional GAN for class-specific generation
            - **SAGAN**: Self-Attention GAN for improved quality
            """)
            
            # Model comparison info
            st.subheader("📊 Model Comparison")
            comparison_data = {
                "Model": ["VAE", "CVAE", "CGAN", "SAGAN"],
                "Latent Dim": [128, 128, 100, 100],
                "Conditional": ["❌", "✅", "✅", "✅"],
                "Quality": ["⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐⭐⭐"]
            }
            st.table(comparison_data)

    # Footer
    st.markdown("---")
    st.markdown(
        "🔬 **Research Project**: Comparing Generative Models on CIFAR-10 | "
        "Built with Streamlit 🎈"
    )

if __name__ == "__main__":
    main()