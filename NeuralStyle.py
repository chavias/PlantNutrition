import streamlit as st
import tensorflow_hub as hub
from PIL import Image
from io import BytesIO
import utils

# Load the style transfer model
hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

st.title('Image Stylization')

st.markdown('### Content Image')
# File uploader for the content image
uploaded_content_file = st.file_uploader("Choose a content image...", type=["jpg", "png", "jpeg","webp"])
if uploaded_content_file is not None:
    content_image = Image.open(uploaded_content_file)
    st.image(content_image, caption=None, use_column_width=True)

st.markdown('### Style Image')
# File uploader for the style image
uploaded_style_file = st.file_uploader("Choose a style image...", type=["jpg", "png", "jpeg","webp"])
if uploaded_style_file is not None:
    style_image = Image.open(uploaded_style_file)
    st.image(style_image, caption=None, use_column_width=True)


if uploaded_content_file is not None and uploaded_style_file is not None:
    # Load the images as tensors, ensuring correct channels
    content_tensor = utils.load_img(content_image)
    style_tensor = utils.load_img(style_image)
    
    # st.write(f"Content Tensor Shape: {content_tensor.shape}")
    # st.write(f"Style Tensor Shape: {style_tensor.shape}")
    
    # Apply the style transfer
    stylized_image = hub_model(content_tensor, style_tensor)[0]
    
    # Convert tensor back to image
    result_image = utils.tensor_to_image(stylized_image)
    
    # print 
    st.markdown('### Stylized Image')
    
    # Display the result
    st.image(result_image, caption=None, use_column_width=True)

    # Provide a download link for the resulting image
    buffered = BytesIO()
    result_image.save(buffered, format="JPEG")
    
    st.download_button(label="Download Stylized Image", data=buffered.getvalue(), file_name="stylized_image.jpg", mime="image/jpeg")
