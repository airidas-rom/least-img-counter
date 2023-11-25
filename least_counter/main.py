import cv2

image_filenames = ["box1.jpg", "box2.jpg", "box3.jpg", "box4.jpg", "box5.jpg", "box6.jpg"]
images = [cv2.imread(filename, cv2.IMREAD_GRAYSCALE) for filename in image_filenames]

total_similarities = [0] * len(image_filenames)  # Initialize total similarities

for template_index, template_image in enumerate(images):
    print(f"Template image: {image_filenames[template_index]}")

    for i, (image, filename) in enumerate(zip(images, image_filenames), start=0):
        result = cv2.matchTemplate(image, template_image, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        print(f"Image {i} ({filename}) similarity:", max_val)

        total_similarities[i] += max_val
    
    print("\n")

least_appeared_index_total = total_similarities.index(min(total_similarities))
least_appeared_image_total = image_filenames[least_appeared_index_total]

print(f"Least appeared image in total: {least_appeared_image_total}")
