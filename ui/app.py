import streamlit as st
from PIL import Image

# Function to calculate similarity
def compare_images(img1, img2):
    # try:
    #     img1 = cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2GRAY)
    #     img2 = cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2GRAY)
    #     img1 = cv2.resize(img1, (500, 500))
    #     img2 = cv2.resize(img2, (500, 500))
    #     score, _ = ssim(img1, img2, full=True)
    #     return score * 100
    # except Exception as e:
    #     st.error("Error in comparison: " + str(e))
        return None

# Streamlit UI
st.title("Flowchart Similarity Checker")

# Instructions
st.write("Upload two flowchart images to compare their similarity. Select the context of the comparison for better insights.")

# Input fields
col1, col2 = st.columns(2)

with col1:
    img1 = st.file_uploader("Upload flowchart image 1", type=["png", "jpg", "jpeg"])
with col2:
    img2 = st.file_uploader("Upload flowchart image 2", type=["png", "jpg", "jpeg"])

category = st.selectbox(
    "Select Comparison Context",
    ["Patent", "Papers", "Reports", "Diagrams"],
    help="Choose the category that best matches the flowcharts being compared."
)

# Compare button
if st.button("Compare"):
    if img1 is not None and img2 is not None:
        try:
            # Load images
            image1 = Image.open(img1)
            image2 = Image.open(img2)
            
            # Compare images
            similarity = compare_images(image1, image2)
            
            if similarity is not None:
                # Display results
                st.write(f"**Similarity Score:** {similarity:.2f}%")
                if similarity > 80:  # Threshold for plagiarism
                    st.error("Verdict: Plagiarized")
                else:
                    st.success("Verdict: Not Plagiarized")
            else:
                st.error("Unable to calculate similarity. Please try again.")
        except Exception as e:
            st.error(f"Error processing images: {e}")
    else:
        st.warning("Please upload both images to proceed.")

# Optional: Display uploaded images
if img1 and img2:
    st.write("Uploaded Images:")
    col1, col2 = st.columns(2)
    with col1:
        st.image(img1, caption="Flowchart Image 1", use_container_width=True)
    with col2:
        st.image(img2, caption="Flowchart Image 2", use_container_width=True)

