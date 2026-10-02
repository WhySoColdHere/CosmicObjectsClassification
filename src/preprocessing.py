from PIL import Image, UnidentifiedImageError
import os


class Preprocessing:
    def __init__(self, data_folder):
        self.data_folder = data_folder

    def get_resized_image(self):
        def resize_image(img_path):
            try:
                img = Image.open(img_path)
                img = img.resize((128, 128))
                return img
            except UnidentifiedImageError:
                print("Error occurred: " + img_path)
                return None

        path = os.path.join('../', self.data_folder)
        image_processed = 0
        errors = 0

        for file in os.listdir(path):
            if os.path.isdir(os.path.join(path, file)):
                for inner_file in os.listdir(os.path.join(path, file)):
                    if inner_file.endswith('.jpg') or inner_file.endswith('.png'):
                        current_img_path = os.path.join(path, file, inner_file)
                        resized_image = resize_image(current_img_path)
                        if resized_image is not None:
                            resized_image.save(current_img_path)
                            image_processed += 1
                        else:
                            errors += 1
                            continue
            elif file.endswith('.jpg') or file.endswith('.png'):
                current_img_path = os.path.join(path, file)
                resized_image = resize_image(current_img_path)
                if resized_image is not None:
                    resized_image.save(current_img_path)
                    image_processed += 1
                else:
                    errors += 1
                    continue

        print("Image processed: " + str(image_processed))
        print("Errors: " + str(errors))


p = Preprocessing('data')
p.get_resized_image()
