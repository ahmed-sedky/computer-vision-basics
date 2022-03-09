import cv2


def show_resized_img(image, scale):
    scale_percent = scale  # percent of original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)

    # resize image
    image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    # show image
    cv2.imshow("image", image)
    cv2.waitKey(0)
