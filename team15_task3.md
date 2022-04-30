# Team 15

| Name | sec | BN |
|------|------|----|
|Ahmed Hossam Mohamed Sedky | 1 | 2 |
|Ahmed Mohammed Abdelfatah | 1 | 5 |
|Ehab Wahba Abdelrahman | 1 | 22 |
|Mo'men Maged Mohammed | 2 | 11 |
|Mohanad Alaa Ragab | 2 | 31 |
----
## Libraries versions
* numpy version **1.21.3**
* cv2 version **4.5.4-dev**
* matplotlib version **3.4.2**

-----
## Code architecture
* Harris
    * Get gradient of the image in x-direction and y-direction using Sobel.
    * Get the second derivative of x,y, and x with respect to y.
    * Apply Gaussian filter to the second derivatives
    * Compute the determine and the trace of this matrix
         $$
            \left(\begin{array}{cc} 
            Ixx & Ixy\\
            Ixy & Iyy
            \end{array}\right)
        $$ 
    * Compute Harris operator "R"
        * where R = determine - k * (trace)^2   , k :corner sharpness = 0.04
* SIFT


* Feature Matching



