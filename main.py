import numpy as np
import matplotlib.pyplot as plt
import cv2


def output_dim(input_size, filter_size, stride):
	dim = (input_size-filter_size)/stride + 1
	return int(dim)


img = cv2.imread('test1.jpg', 0)
img = cv2.resize(img, (100, 100))

FILTER_SIZE = 3
STRIDE = 1
img_row, img_col = img.shape

filter_vals = [0, -1, -1, 
			   -1, 0,  -1, 
			  -1, -1,  0]

output = []

for i in range(0, img_row-FILTER_SIZE+1, STRIDE):
	rows = [i+x for x in range(FILTER_SIZE)]
	# print(rows)

	for j in range(0, img_col-FILTER_SIZE+1, STRIDE):
		cols = [j+x for x in range(FILTER_SIZE)]
		# print(cols)

	# print()

		img_vector = []
		for row in rows:
			for col in cols: 
				pixel = img[row][col] 
				img_vector.append(pixel)
		output_pixel = np.sum(np.multiply(img_vector, filter_vals))
		output.append(output_pixel)


output_row_dim = output_dim(img_row, FILTER_SIZE, STRIDE)
output_col_dim = output_dim(img_col, FILTER_SIZE, STRIDE)

output = np.array(output)
print(type(output))
# print(output)
output_img = output.reshape(output_row_dim, output_col_dim)


# print(len(output))
ax1 = plt.subplot(1, 2, 1)
ax1.imshow(img, cmap='gray')
plt.title('Original Image')
ax2 = plt.subplot(1, 2, 2)
ax2.imshow(output_img, cmap='gray')
plt.title('Filtered Image')


plt.show()


print('End..!')