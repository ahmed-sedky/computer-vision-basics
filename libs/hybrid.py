from libs import filters,Sobel
def hybrid (image1 ,image2):
    gaussian_blurred_image = filters.gaussian_filter(image1,21,3)
    sobel_img = Sobel.sobel(image2)
    hybrid_img = sobel_img + gaussian_blurred_image
    return hybrid_img
