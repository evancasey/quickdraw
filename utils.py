import scipy
import scipy.misc


def save_img(dir, name, img):
    scipy.misc.imsave("{0}/{1}.jpg".format(dir, name), img)
