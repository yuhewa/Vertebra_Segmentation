import os
from cv2 import cv2
import numpy as np

# # 因為rucursive會overflow, 改用下面的函式
# def floodfill(matrix, x, y, color):
#     if matrix[x][y] == 255:  
#         matrix[x][y] = color 
#         #recursively invoke flood fill on all surrounding cells:
#         if x > 0:
#             floodfill(matrix,x-1,y,color)
#         if x < len(matrix[y]) - 1:
#             floodfill(matrix,x+1,y,color)
#         if y > 0:
#             floodfill(matrix,x,y-1,color)
#         if y < len(matrix) - 1:
#             floodfill(matrix,x,y+1,color)


# Pure Python, usable speed but over 10x greater runtime than Cython version
def fill(data, start_coords, fill_value):

    xsize, ysize = data.shape
    orig_value = data[start_coords[0], start_coords[1]]
    stack = set(((start_coords[0], start_coords[1]),))

    while stack:
        x, y = stack.pop()

        if data[x, y] == orig_value:
            data[x, y] = fill_value
            if x > 0:
                stack.add((x - 1, y))
            if x < (xsize - 1):
                stack.add((x + 1, y))
            if y > 0:
                stack.add((x, y - 1))
            if y < (ysize - 1):
                stack.add((x, y + 1))




root_dir = os.getcwd() 
label_dir = os.path.join(root_dir, 'Final_test', 'origin_label')
label_processed_dir = os.path.join(root_dir,'Final_test','label')
filenames = os.listdir(label_dir)



# for filename in filenames:
#     img = cv2.imread( os.path.join(label_dir,'filename') )
#     cv2.imshow("img",img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#     break


for k in range(4):
    color = 1
    mask = cv2.imread( os.path.join(label_dir,filenames[k]),  cv2.IMREAD_GRAYSCALE)
    for i in range(1200):
        for j in range(500):
            if mask[i,j] == 255:
                fill(mask,(i,j),color)
                color += 1
    name = '000'+str(k+1)+'.png'
    cv2.imwrite(os.path.join(label_processed_dir,name) , mask)






# masks = np.zeros( (np.max(mask), mask.shape[0], mask.shape[1] ), dtype=np.uint8)
# obj_ids = np.unique(mask)
# obj_ids = obj_ids[1:]

# # 取出各個物件的mask
# num_objs = len(obj_ids)
# for k in range(num_objs):
#     for i in range(mask.shape[0]):
#         for j in range(mask.shape[1]):
#             if (mask[i][j] == obj_ids[k]):
#                 masks[k][i][j] = True
          
# # 觀看取出的mask
# for i in range(num_objs):
#     cv2.imshow("img",masks[i])
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()



