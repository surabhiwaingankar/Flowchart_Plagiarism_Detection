import streamlit as st
from PIL import Image
import os
from datetime import datetime
from functionality.predict import inspect_raw_detections
from functionality.utils.visualize_detection import visualize_prediction
from functionality.utils.shapes_arrows_separation import separate_shapes_and_arrows
from functionality.utils.match_text_to_shape import shape_text_matching
from functionality.utils.arrow_analysis import enhanced_arrow_connection_analysis
from functionality.graph import final_graph_structure
from functionality.overall_similarity import graph_similarity
from functionality.utils.graph_builder import build_graph


# Function to calculate similarity
def compare_images(img1_path, img2_path):
    try:
        # Perform detection
        detections1, res_img1 = inspect_raw_detections(img1_path)
        detections2, res_img2 = inspect_raw_detections(img2_path)

        # Visualize prediction
        visualize_prediction(res_img1)
        visualize_prediction(res_img2)

        # Separate shapes and arrows
        shapes1, arrows1 = separate_shapes_and_arrows(detections1)
        shapes2, arrows2 = separate_shapes_and_arrows(detections2)

        # Arrange shapes
        sorted_shapes1 = shape_text_matching(img1_path, shapes1)
        sorted_shapes2 = shape_text_matching(img2_path, shapes2)

        # Analyze arrows
        adjacency1, shape_details1, sorted_shapes1 = enhanced_arrow_connection_analysis(
            sorted_shapes1, arrows1
        )
        adjacency2, shape_details2, sorted_shapes2 = enhanced_arrow_connection_analysis(
            sorted_shapes2, arrows2
        )

        # Build graph
        shapes1, edges1 = final_graph_structure(
            adjacency1, shape_details1, sorted_shapes1
        )
        shapes2, edges2 = final_graph_structure(
            adjacency2, shape_details2, sorted_shapes2
        )

        graph1 = build_graph(shapes1, edges1)
        graph2 = build_graph(shapes2, edges2)

        # Calculate similarity
        similarity = graph_similarity(graph1, graph2)

        return similarity

    except Exception as e:
        st.error("Error in comparison: " + str(e))
        return None


# Define a directory to save uploaded images
UPLOAD_DIR = "uploaded_images"

# Create the directory if it doesn't exist
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# Streamlit UI
st.title("Flowchart Similarity Checker")

# Instructions
st.write("A tool to compare flowcharts and determine if they are plagiarized.")

# Input fields
col1, col2 = st.columns(2)

with col1:
    img1 = st.file_uploader("Upload flowchart image 1", type=["png", "jpg", "jpeg"])
with col2:
    img2 = st.file_uploader("Upload flowchart image 2", type=["png", "jpg", "jpeg"])

category = st.selectbox(
    "Select Comparison Context",
    ["Patent", "Research Papers", "Book Publishing", "Reports"],
    help="Choose the category that best matches the flowcharts being compared.",
)

# Define thresholds for each category
thresholds = {
    "Patent": 90,  # High similarity threshold for patents
    "Research Papers": 80,  # Moderate similarity threshold for research papers
    "Book Publishing": 70,  # Lower threshold for book publishing
    "Reports": 60,  # Lowest threshold for reports
}

# Compare button
if st.button("Compare"):
    if img1 is not None and img2 is not None:
        try:
            # Load images
            image1 = Image.open(img1)
            image2 = Image.open(img2)

            # Save images to the defined directory with unique filenames
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            img1_path = os.path.join(UPLOAD_DIR, f"flowchart1_{timestamp}.png")
            img2_path = os.path.join(UPLOAD_DIR, f"flowchart2_{timestamp}.png")

            image1.save(img1_path)
            image2.save(img2_path)

            # Compare images
            similarity = compare_images(img1_path, img2_path)

            if similarity is not None:
                # Get the threshold for the selected category
                threshold = thresholds[category]

                # Display results
                st.write(f"**Similarity Score:** {similarity:.2f}%")
                if similarity >= threshold:  # Check against category-specific threshold
                    st.error(f"Verdict: Plagiarized (Threshold: {threshold}%)")
                else:
                    st.success(f"Verdict: Not Plagiarized (Threshold: {threshold}%)")
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
