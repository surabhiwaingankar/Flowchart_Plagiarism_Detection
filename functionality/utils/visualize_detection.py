import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def visualize_prediction(image_path):
    img = mpimg.imread(image_path)

    plt.figure(figsize=(5, 5))
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Pred: {image_path}", fontsize=8)
    plt.tight_layout()
    plt.show()
