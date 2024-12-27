import cv2
import numpy as np

def generate_chessboard(rows, cols, square_size, border_size):
    chessboard = np.zeros((rows * square_size, cols * square_size), dtype=np.uint8)
    for i in range(rows):
        for j in range(cols):
            if (i + j) % 2 == 0:
                chessboard[i * square_size:(i + 1) * square_size, j * square_size:(j + 1) * square_size] = 255
    
    # 在棋盘周围添加白色边框
    chessboard_with_border = cv2.copyMakeBorder(
        chessboard,
        top=border_size,
        bottom=border_size,
        left=border_size,
        right=border_size,
        borderType=cv2.BORDER_CONSTANT,
        value=255
    )
    return chessboard_with_border

rows = 6
cols = 9
square_size = 50
border_size = 50
chessboard = generate_chessboard(rows, cols, square_size, border_size)

cv2.imwrite('chessboard.png', chessboard)
cv2.imshow('Chessboard', chessboard)
cv2.waitKey(0)
cv2.destroyAllWindows()