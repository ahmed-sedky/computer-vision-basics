from libs.utils import max_min
def normalize_RGB(image , new_max ,new_min):
    min_b,max_b,min_g,max_g,min_r,max_r =  max_min(image)
    width, height = image.size
    pixels = image.load()
    for i in range(width):
        for j in range(height):
            pixels[i, j] = (int(
                ((pixels[i, j][0] - min_b) * ((new_max - new_min) / (max_b - min_b)))
                + new_min
            ), int(
                ((pixels[i, j][1] - min_g) * ((new_max - new_min) / (max_g - min_g)))
                + new_min
            ), int(
                ((pixels[i, j][2] - min_r) * ((new_max - new_min) / (max_r - min_r)))
                + new_min
            ))

def normalize_Gray(image , new_max ,new_min):
    min_gray,max_gray =  max_min(image)
    width, height = image.size
    pixels = image.load()
    for i in range(width):
        for j in range(height):
            pixels[i,j] = int ( ( (pixels[i,j] - min_gray) * ( (new_max - new_min) /  (max_gray - min_gray) ) ) + new_min)

def normalize(image ,new_max ,new_min):
    if image.mode == 'L':
        normalize_Gray(image,new_max,new_min)
    else:
        normalize_RGB(image,new_max,new_min)
