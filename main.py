from skimage.filters import median
from skimage.morphology import ball
from image_utils import load_image, edge_detection

image = load_image('/content/1268.jpg')
clean_image = median(image, ball(3))
edgeMAG = edge_detection(clean_image)

plt.hist(edgeMAG.ravel(), bins=256)
plt.title("Histogram of Edge Magnitude")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.show()

threshold = 50
edge_binary = edgeMAG > threshold

plt.imshow(edge_binary, cmap='gray')
plt.title("Binary Edge-Detected Image")
plt.axis('off')
plt.show()


edge_image = Image.fromarray((edge_binary * 255).astype(np.uint8)) 
edge_image.save("my_edges.png")
print("Binary edge-detected image saved as 'my_edges.png'.")
