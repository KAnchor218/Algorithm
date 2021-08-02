import pygame
from pygame.locals import *
import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


class Brush():
    def __init__(self, screen):
        self.screen = screen
        self.color = (255, 255, 255)
        self.size = 10
        self.drawing = False
        self.last_pos = None  # <--

    def start_draw(self, pos):
        self.drawing = True
        self.last_pos = pos  # <--

    def end_draw(self):
        self.drawing = False

    def draw(self, pos):
        if self.drawing:
            for p in self._get_points(pos):
                # draw eveypoint between them
                pygame.draw.circle(self.screen,
                                   self.color, p, self.size)

    def _get_points(self, pos):
        """ Get all points between last_point ~ now_point. """
        points = [(self.last_pos[0], self.last_pos[1])]
        len_x = pos[0] - self.last_pos[0]
        len_y = pos[1] - self.last_pos[1]
        length = math.sqrt(len_x ** 2 + len_y ** 2)
        step_x = len_x / length
        step_y = len_y / length
        for i in range(int(length)):
            points.append(
                (points[- 1][0] + step_x, points[- 1][1] + step_y))
        points = map(lambda x: (int(0.5 + x[0]), int(0.5 + x[1])), points)
        # return light-weight, uniq list
        self.last_pos = pos
        return list(set(points))


class Painter():
    def __init__(self, path):
        self.screen = pygame.display.set_mode((280, 280))
        pygame.display.set_caption("Painter")
        self.clock = pygame.time.Clock()
        self.brush = Brush(self.screen)
        self.path = path
        self.count = 0

    def run(self):
        self.screen.fill((0, 0, 0))
        while True:
            # max fps limit
            self.clock.tick(30)
            for event in pygame.event.get():
                if event.type == QUIT:
                    return
                elif event.type == KEYDOWN:
                    if event.key == K_ESCAPE:
                        self.screen.fill((0, 0, 0))
                    if event.key == K_RETURN:
                        pygame.image.save(self.screen, self.path + str(self.count) + '.jpg')
                        self.screen.fill((0, 0, 0))
                        self.count += 1
                        return
                    if event.key == K_SPACE:
                        pygame.image.save(self.screen, self.path + str(self.count) + '.jpg')
                        self.screen.fill((0, 0, 0))
                        self.count += 1
                elif event.type == MOUSEBUTTONDOWN:
                    self.brush.start_draw(event.pos)
                elif event.type == MOUSEMOTION:
                    self.brush.draw(event.pos)
                elif event.type == MOUSEBUTTONUP:
                    self.brush.end_draw()

            pygame.display.update()


def load_preprosess_image(path):
    img_raw = tf.io.read_file(path)
    img_tensor = tf.cast(tf.image.decode_jpeg(img_raw, channels=1), tf.float32)
    img_tensor = tf.image.resize(img_tensor, [28, 28])
    img = img_tensor / 255
    return img


if __name__ == '__main__':
    app = Painter('./img/')
    app.run()
    ID = []
    for count in range(app.count):
        img = load_preprosess_image('./img/' + str(count) + '.jpg')
        # plt.imshow(img, cmap='gray')
        # plt.show()
        np.expand_dims(img, axis=-1)

        model = tf.keras.models.load_model('./model.h5')
        prediction = model.predict(img.numpy().reshape(1, 28, 28, 1))
        result = np.argmax(prediction)
        ID.append(result)
    print('学号为：', end='')
    for item in ID:
        print(item, end='')
