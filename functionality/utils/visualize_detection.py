import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

def visualize_predictions(dataset_path):
    results_path = f'{dataset_path}/runs/detect/predict'
    images = [f for f in os.listdir(results_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    plt.figure(figsize=(20, 15))
    for idx, img_name in enumerate(images[:6]):
        img_path = os.path.join(results_path, img_name)
        img = mpimg.imread(img_path)
        plt.subplot(2, 3, idx+1)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Pred: {img_name}", fontsize=8)
    plt.tight_layout()
    plt.show()