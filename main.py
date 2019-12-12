import time
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
print()

def output_dim(input_size, filter_size, stride):
	dim = (input_size-filter_size)/stride + 1
	return int(dim)


def input_filter_val(f):
	try:
		val = int(input(f'f{f+1} : '))
	except:
		print('Invalid Val. Try Again..')
		val = input_filter_val(f)
	return val


# Importing the Image
img = cv2.imread('test1.jpg', 0)
# img = cv2.resize(img, (100, 100)) # Resizing temporarily for faster calculations

# PARAMETERS
print('------- PARAMETERS -------')
STRIDE      = int(input('Enter Stride      : '))
FILTER_SIZE = int(input('Enter Filter Size : '))
img_row, img_col = img.shape

print()
print('---- Filter Variables ----')
for i in range(FILTER_SIZE):
  for j in range(FILTER_SIZE):
    print(f' f{(i*3)+j+1} ', end='')
  print()


print()
print('--- Input Filter Values ---')
# Filter
filter_vals = []

for i in range(FILTER_SIZE**2):
	val = input_filter_val(i)
	filter_vals.append(val)

print()

output_row_dim = output_dim(img_row, FILTER_SIZE, STRIDE)
output_col_dim = output_dim(img_col, FILTER_SIZE, STRIDE)

total_pixels = output_row_dim * output_col_dim


output = []
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




output = np.array(output)
output_img = output.reshape(output_row_dim, output_col_dim)

print(f'Time Taken : {round(time.time()-start, 2)}s')

ax1 = plt.subplot(1, 2, 1)
ax1.imshow(img, cmap='gray')
plt.title('Original Image')
ax2 = plt.subplot(1, 2, 2)
ax2.imshow(output_img, cmap='gray')
plt.title('Filtered Image')

plt.show()


print('End..!')
