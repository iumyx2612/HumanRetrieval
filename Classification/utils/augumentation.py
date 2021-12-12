import albumentations as A
import cv2
import os
import matplotlib.pyplot as plt

if __name__ == '__main__':
    transform = A.ColorJitter(brightness=(1, 1), contrast=(1.3, 1.3), saturation=(1, 1), hue=(0, 0), p=1)
    for _type in os.listdir("../Dataset/Train"):
        count = 0
        _type_path = os.path.join("../Dataset/Train", _type)
        if os.path.isdir(_type_path):
            for image in os.listdir(_type_path):
                if "green" in image:
                    count += 1
                    image_name = image.split('.')[0]
                    img_path = os.path.join(_type_path, image)
                    img = cv2.imread(img_path)[:, :, ::-1]
                    new_img = transform(image=img)["image"]
                    #new_img = cv2.cvtColor(new_img, cv2.COLOR_HSV2RGB)
                    plt.imshow(new_img)
                    plt.title(image_name)
                    plt.show()
    print(count)