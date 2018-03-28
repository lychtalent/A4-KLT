import numpy as ny
from scipy import interpolate as intplt
from scipy import ndimage as ng
from numpy.linalg import inv
import cv2

def grad(image):
  scharr_x = ny.array([[3,0,-3],[10,0,-10],[3,0,-3]])
  scharr_y = ny.array([[3,10,3],[0,0,0],[-3,-10,-3]])

  image_x = ng.convolve(image,scharr_x,mode='reflect')
  image_y = ng.convolve(image,scharr_y,mode='reflect')
  return image_x,image_y

def window(image,centre,window_size=[51,51]):
  sub = image[centre[0]-(window_size[0]-1)/2:centre[0]+(window_size[0]-1)/2+1,centre[1]-(window_size[1]-1)/2:centre[1]+(window_size[1]-1)/2+1]
  return sub

def estimate_z(dx,dy):
  z_x = ny.square(dx).sum()
  z_y = ny.square(dy).sum()
  z_xy = ny.multiply(dx,dy).sum()
  z = [[z_x,z_xy],[z_xy,z_y]]
  return z

def diff_fun(imagef,imageg,dx,dy):
  diff = imageg-imagef

  e_x = ny.multiply(diff,dx)
  e_y = ny.multiply(diff,dy)
  e = [[e_x.sum()],[e_y.sum()]]
  return e

def interp_img(image,centre,h,size=[51,51]):
  out = ny.zeros(size)
  minx = centre[0] - (size[0]-1)/2 + h[0,0]
  maxx = centre[0] + (size[0]-1)/2 + h[0,0]
  miny = centre[1] - (size[1]-1)/2 + h[0,1]
  maxy = centre[1] + (size[1]-1)/2 + h[0,1]
  x = ny.arange(minx,maxx+1)
  y = ny.arange(miny,maxy+1)
  r,c = ny.shape(image)
  row = ny.arange(r)
  col = ny.arange(c)
  f = intplt.interp2d(col,row,image,kind="cubic")
  out = f(y,x)
  return out

itern = 300
centre = [320,336]
centre_true = [289.35,336.12]
h = ny.matrix([[0.001,0.001]])


# load images
image1 = cv2.imread("view0.png")
image2 = cv2.imread("view1.png")
image1 = cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)
image2 = cv2.cvtColor(image2,cv2.COLOR_BGR2GRAY)

image_x, image_y = grad(image1)
image2_x,image2_y = grad(image2)
image2_window = window(image2,centre)

image1_window = window(image1,centre)

sub2_dx = window(image2_x,centre)
sub2_dy = window(image2_y,centre)
mis = image2_window-image1_window
mis = ny.sum(ny.square(mis))
mis = mis**(0.5)
print("start mishit: ", mis)
image_show = cv2.imread("view0.png")
image_show = cv2.circle(image_show,(ny.int(centre_true[0]),ny.int(centre_true[1])),ny.int(40),(60,60,60),2)
cv2.imshow("rec",image_show)
cv2.waitKey(0)
for i in ny.arange(itern):

  sub_dx = interp_img(image_x,centre,h)
  sub_dy = interp_img(image_y,centre,h)
  image1_window = interp_img(image1,centre,h)

  z = estimate_z(sub_dx,sub_dy)
  e = diff_fun(image1_window,image2_window,sub_dx,sub_dy)
  h_new = ny.matmul(inv(z),e)
  h = ny.add(h,h_new.transpose())
  print("h ",h)
  image1_window = interp_img(image1,centre,h)
  mis = image2_window-image1_window
  mis = ny.sum(ny.square(mis))
  mis = mis**(0.5)
  print("mishit: ", mis)
  if i%30 == 0:
    image_show = cv2.circle(image_show,(ny.int(centre[0]+h[0,0]),ny.int(centre[1]+h[0,1])),ny.int(40),ny.random.randint(255,size=3),2)
    cv2.imshow("rec",image_show)
    cv2.waitKey(0)
  if ny.sum(ny.square(h_new)) <= 0.0001:
    print("Movement less than 0.01")
    break
print("Reached max interation")
cv2.destroyAllWindows()

