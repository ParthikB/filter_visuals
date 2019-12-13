import time
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import urllib.request
print()

# Computing the dimensions of the Output Image
def output_dim(input_size, filter_size, stride):
	dim = (input_size-filter_size)/stride + 1
	return int(dim)


# Function to input Filter Vals
def input_filter_val(f):
	try:
		val = int(input(f'f{f+1} : '))
	except:
		print('Invalid Val. Try Again..')
		val = input_filter_val(f)
	return val


# Function to use an Image from the Internet
def urlToImage(url):
  resp = urllib.request.urlopen(url)
  img = np.asarray(bytearray(resp.read()), dtype='uint8')
  img = cv2.imdecode(img, cv2.IMREAD_COLOR)
  original = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  return original, img
 
##############################################################################

# Importing the Image
print('(Leave blank to use default image)')
URL = input('Enter the URL of the Image : ')
if not URL:
  original = cv2.imread('dummy.jpg') 
  img      = cv2.imread('dummy.jpg', 0) 
else:
  original, img = urlToImage(URL)
print()


# PARAMETERS
print('------- PARAMETERS -------')
STRIDE      = int(input('Enter Stride      : '))
FILTER_SIZE = int(input('Enter Filter Size : '))
img_row, img_col = img.shape
print()

# Creating a Filter Visual Template
print('---- Filter Variables ----')
for i in range(FILTER_SIZE):
  for j in range(FILTER_SIZE):
    print(f' f{(i*3)+j+1} ', end='')
  print()
print()

# Input the Filter Values
print('--- Input Filter Values ---')
filter_vals = [] # Blank list to append the Filter Values into

for i in range(FILTER_SIZE**2):
	val = input_filter_val(i)
	filter_vals.append(val)
print()


# Filtering the Output Image
output = [] # Array to append the Output Image pixels to
print('Processing Image...')
start = time.time()

for i in tqdm(range(0, img_row-FILTER_SIZE+1, STRIDE)):
	rows = [i+x for x in range(FILTER_SIZE)]

	for j in range(0, img_col-FILTER_SIZE+1, STRIDE):
		cols = [j+x for x in range(FILTER_SIZE)]

		img_vector = []
		for row in rows:
			for col in cols:
				pixel = img[row][col] 
				img_vector.append(pixel)

		output_pixel = np.sum(np.multiply(img_vector, filter_vals))
		output.append(output_pixel)


# Computing the Output Image Dimensions
output_row_dim = output_dim(img_row, FILTER_SIZE, STRIDE)
output_col_dim = output_dim(img_col, FILTER_SIZE, STRIDE)

# Converting the Output List into Numpy Array for reshaping
output = np.array(output)
output_img = output.reshape(output_row_dim, output_col_dim)

print(f'Time Taken : {round(time.time()-start, 2)}s')

# PLOTTING
ax1 = plt.subplot(1, 2, 1)
ax1.imshow(img)
plt.title('Original Image')
ax2 = plt.subplot(1, 2, 2)
ax2.imshow(output_img, cmap='gray')
plt.title('Filtered Image')
plt.show()
plt.savefig('output.png')

print("Output saved. Check 'Output.png'!")
