import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image

# Title
st.title("Stable Diffusion Text-to-Image Generator")
st.write("Type a prompt and click Generate to create an image.")        

# Load Model
@st.cache_resource(show_spinner=True)
def load_model():
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id)
    pipe = pipe.to("cuda") if torch.cuda.is_available() else pipe.to("cpu")
    return pipe

pipe = load_model()

# User Input Prompt
prompt = st.text_input("Enter your prompt:")
generate = st.button("Generate")

if generate:
    with st.spinner("Generating image..."):
        image = pipe(prompt).images[0]
        st.image(image, caption="Generated Image", use_container_width=True)
        # Download button
        with st.expander("Download Image"):
            import io
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            st.download_button(
                label="Download as PNG",
                data=buf.getvalue(),
                file_name="generated_image.png",
                mime="image/png"
            )