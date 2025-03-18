import os
import cv2
import multiprocessing

def check_and_delete_image(image_path: str, min_width: int, min_height: int):
    """Check image size and delete if smaller than min_width x min_height."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Skipping corrupted image: {image_path}")
        return 0  # No deletion

    height, width, _ = image.shape
    if width < min_width or height < min_height:
        os.remove(image_path)
        # print(f"Deleted: {image_path} (Size: {width}x{height})")
        return 1  # Deletion happened

    return 0  # No deletion

def delete_small_images_parallel(image_dir: str, min_width: int, min_height: int, num_workers: int = 8):
    """
    Parallelized deletion of images smaller than `min_width` x `min_height`.

    Args:
        image_dir (str): Directory containing images.
        min_width (int): Minimum required width.
        min_height (int): Minimum required height.
        num_workers (int): Number of processes to use.
    """
    image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    if not image_paths:
        print("No valid images found in the directory.")
        return

    # Use multiprocessing pool
    with multiprocessing.Pool(num_workers) as pool:
        results = pool.starmap(check_and_delete_image, [(path, min_width, min_height) for path in image_paths])

    print(f"Total images deleted: {sum(results)}")

if __name__ == "__main__":
    # Set parameters
    image_directory = r"E:\Manga109"  # Change this to your image directory
    min_w, min_h = 256, 256  # Set the minimum width and height
    num_processes = multiprocessing.cpu_count()  # Use all available CPU cores

    # Run the function
    delete_small_images_parallel(image_directory, min_w, min_h, num_processes)
