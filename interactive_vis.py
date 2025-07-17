import streamlit as st
import torch
from data_prep import get_cifar10
from vae_model import VAE
from visualization import interpolate_latent_space, visualize_latent_space
from utils import load_checkpoint
import os

def load_model(model, checkpoint_path, device, optimizer=None):
    if os.path.exists(checkpoint_path):
        if optimizer is None:
            model.load_state_dict(torch.load(checkpoint_path, map_location=device)['model_state_dict'])
        else:
            load_checkpoint(model, optimizer, checkpoint_path, device)
        st.write(f"Loaded {checkpoint_path}")
    else:
        st.error(f"Checkpoint {checkpoint_path} not found!")
    model.eval()
    return model

def main():
    st.markdown("<h1 style='text-align: center; color: white;'>Interactive Latent Space Visualization for Generative Models</h1>", unsafe_allow_html=True)

    #st.title("Interactive Latent Space Visualization for Generative Models")
    st.write("Explore how latent space values influence generated CIFAR-10 images for VAE, GAN, and Diffusion models.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st.write(f"Using device: {device}")

    _, test_loader, _ = get_cifar10(batch_size=64)

    vae = VAE(latent_dim=128).to(device)

    vae_checkpoint = "checkpoints/vae_final.pth"
    

    vae = load_model(vae, vae_checkpoint, device)

    model_choice = st.selectbox("Select Model", ["VAE"])

    if model_choice in ["VAE"]:
        viz_type = st.radio("Visualization Type", ["Interpolation", "Latent Space Scatter (t-SNE/PCA)"])
        model = vae 
        latent_dim = 128 

        if viz_type == "Interpolation":
            st.write("Interpolate between two latent vectors to see smooth transitions in generated images.")
            n_steps = st.slider("Number of Interpolation Steps", 5, 20, 10)
            if st.button("Generate Interpolation"):
                img = interpolate_latent_space(model, test_loader, n_steps=n_steps, device=device, streamlit=True)
                st.image(img, caption="Latent Space Interpolation", use_container_width=True)


        elif viz_type == "Latent Space Scatter (t-SNE/PCA)":
            st.write("Visualize the latent space in 3D using t-SNE or PCA, colored by CIFAR-10 class labels.")
            method = st.selectbox("Dimensionality Reduction Method", ["t-SNE", "PCA"])
            n_samples = st.slider("Number of Samples", 100, 5000, 1000)
            if st.button("Generate Scatter Plot"):
                fig = visualize_latent_space(model, test_loader, method=method.lower(), n_samples=n_samples, device=device, streamlit=True)
                st.pyplot(fig)

    

if __name__ == "__main__":
    main()