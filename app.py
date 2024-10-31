import pygame
import sys
from pygame.locals import *
import numpy as np
from keras.models import load_model
import cv2

# Set up window dimensions
WINDOWSIZEX = 640
WINDOWSIZEY = 480

BOUNDARYINC = 5
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

IMAGESAVE = False

MODEL = load_model("bestmodel.keras")

LABELS = {0: "Zero", 1: "One", 
          2: "Two", 3: "Three", 
          4: "Four", 5: "Five", 
          6: "Six", 7: "Seven", 
          8: "Eight", 9: "Nine"}

# Initialize pygame
pygame.init()

# Set up font
FONT = pygame.font.Font(None, 36)

# Set up display
DISPLAYSURF = pygame.display.set_mode((WINDOWSIZEX, WINDOWSIZEY))

pygame.display.set_caption('Digit Board')

iswriting = False

number_xcord = []
number_ycord = []

img_cnt = 1

PREDICT = True

# Main loop
while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()

        if event.type == MOUSEMOTION and iswriting:
            xcord, ycord = event.pos
            pygame.draw.circle(DISPLAYSURF, WHITE, (xcord, ycord), 4, 0)
            number_xcord.append(xcord)
            number_ycord.append(ycord)

        if event.type == MOUSEBUTTONDOWN:
            iswriting = True

        if event.type == MOUSEBUTTONUP:
            iswriting = False
            number_xcord = sorted(number_xcord)
            number_ycord = sorted(number_ycord)

            # Define bounding box
            rect_min_x = max(number_xcord[0] - BOUNDARYINC, 0)
            rect_max_x = min(WINDOWSIZEX, number_xcord[-1] + BOUNDARYINC)
            rect_min_y = max(number_ycord[0] - BOUNDARYINC, 0)
            rect_max_y = min(WINDOWSIZEY, number_ycord[-1] + BOUNDARYINC)

            number_xcord = []
            number_ycord = []

            # Get the drawn image within bounding box
            img_arr = np.array(pygame.PixelArray(DISPLAYSURF))[rect_min_x:rect_max_x, rect_min_y:rect_max_y].T.astype(np.float32)

            if IMAGESAVE:
                cv2.imwrite(f"image_{img_cnt}.png", img_arr)  # Save the image correctly
                img_cnt += 1

            if PREDICT:
                # Preprocess the image for prediction
                image = cv2.resize(img_arr, (28, 28))
                image = np.pad(image, (10, 10), 'constant', constant_values=0)
                image = cv2.resize(image, (28, 28)) / 255.0

                # Predict the digit
                prediction = MODEL.predict(image.reshape(1, 28, 28, 1))
                label = str(LABELS[np.argmax(prediction)])

                # Display prediction
                textSurface = FONT.render(label, True, RED, WHITE)
                textRectObj = textSurface.get_rect()
                textRectObj.left, textRectObj.bottom = rect_min_x, rect_min_y
                DISPLAYSURF.blit(textSurface, textRectObj)

        if event.type == KEYDOWN:
            if event.unicode == "n":
                DISPLAYSURF.fill(BLACK)  # Clear the screen

        pygame.display.update()

    


