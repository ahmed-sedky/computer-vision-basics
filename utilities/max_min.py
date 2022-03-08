def max_min(img):
    if len(img.shape) == 3:
        rows,cols,_ = img.shape 
        min_b = 1000 ; max_b = 0 ; min_g =1000 ; max_g =0 ; min_r =1000 ; max_r =0
        for i in range(rows):
            for j in range(cols):
                if img[i,j][2] < min_r:
                    min_r = img[i,j][2]
                if img[i,j][1] < min_g:
                    min_g = img[i,j][1]
                if img[i,j][0] < min_b:
                    min_b = img[i,j][0]

                if img[i,j][2] > max_r:
                    max_r = img[i,j][2]
                if img[i,j][1] > max_g:
                    max_g = img[i,j][1]
                if img[i,j][0] > max_b:
                    max_b = img[i,j][0]
        return min_b,max_b,min_g,max_g,min_r,max_r
    else : 
        shape =img.shape
        min_gray = 1000 ; max_gray = 0 
        for i in range(shape[0]):
            for j in range(shape[1]):
                if img[i,j] < min_gray:
                    min_gray = img[i,j]
                if img[i,j] > max_gray:
                    max_gray = img[i,j]
        return min_gray,max_gray