
# Divid the image with image_width into grids and apply the
# given operation.
#
# The image is divided into multiple grids of different sizes.
# Starting from width w, w/2, w/4 to the min_width provided.
# The function operator will be called with the subimage as
# the parameter.
def apply_to_zones(image, image_width, min_width, operator):
    w = image_width
    while(w >= min_width):
        half_w = w/2
        for i in xrange(0, image_width, w):
            for j in xrange(0, image_width, w):
                subimage = image[i:i+w, j:j+w]
                operator(subimage)
        w = half_w
