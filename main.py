import numpy as np
from vis_nav_game import Player, Action, Phase
import pygame
import cv2
import os
import pickle
from sklearn.cluster import KMeans
from sklearn.neighbors import BallTree
from odometry import Odometry


class KeyboardPlayerPyGame(Player):
    def __init__(self):
        super(KeyboardPlayerPyGame, self).__init__()
        self.fpv = None
        self.last_act = Action.IDLE
        self.screen = None
        self.keymap = None
        self.direction = ""
        self.odom = None

        self.count = 0
        self.save_dir = "data/images/"
        self.sift = cv2.SIFT_create()
        # self.codebook = pickle.load(open("data/codebook.pkl", "rb")
        self.database = []
        self.positions = []
        

    def reset(self):
        self.fpv = None
        self.last_act = Action.IDLE
        self.screen = None

        pygame.init()

        self.keymap = {
            pygame.K_LEFT: Action.LEFT,
            pygame.K_RIGHT: Action.RIGHT,
            pygame.K_UP: Action.FORWARD,
            pygame.K_DOWN: Action.BACKWARD,
            pygame.K_SPACE: Action.CHECKIN,
            pygame.K_ESCAPE: Action.QUIT
        }

    def act(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                self.last_act = Action.QUIT
                return Action.QUIT
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    self.direction = "left"
                    self.last_act = Action.LEFT
                elif event.key == pygame.K_RIGHT:
                    self.direction = "right"
                    self.last_act = Action.RIGHT
                elif event.key == pygame.K_UP:
                    self.direction = "forward"
                    self.last_act = Action.FORWARD
                elif event.key == pygame.K_DOWN:
                    self.direction = "reverse"
                    self.last_act = Action.BACKWARD
                elif event.key == pygame.K_ESCAPE:
                    self.last_act = Action.QUIT
                    self.odom.reset_position()
            else:
                self.direction = ""
                self.last_act = Action.IDLE

        return self.last_act

    def show_target_images(self):
        targets = self.get_target_images()
        if targets is None or len(targets) <= 0:
            return
        hor1 = cv2.hconcat(targets[:2])
        hor2 = cv2.hconcat(targets[2:])
        concat_img = cv2.vconcat([hor1, hor2])

        w, h = concat_img.shape[:2]
        
        color = (0, 0, 0)

        concat_img = cv2.line(concat_img, (int(h/2), 0), (int(h/2), w), color, 2)
        concat_img = cv2.line(concat_img, (0, int(w/2)), (h, int(w/2)), color, 2)

        w_offset = 25
        h_offset = 10
        font = cv2.FONT_HERSHEY_SIMPLEX
        line = cv2.LINE_AA
        size = 0.75
        stroke = 1

        cv2.putText(concat_img, 'Front View', (h_offset, w_offset), font, size, color, stroke, line)
        cv2.putText(concat_img, 'Left View', (int(h/2) + h_offset, w_offset), font, size, color, stroke, line)
        cv2.putText(concat_img, 'Back View', (h_offset, int(w/2) + w_offset), font, size, color, stroke, line)
        cv2.putText(concat_img, 'Right View', (int(h/2) + h_offset, int(w/2) + w_offset), font, size, color, stroke, line)

        cv2.imshow(f'KeyboardPlayer:target_images', concat_img)
        cv2.imwrite('target.jpg', concat_img)
        cv2.waitKey(1)
        
        def convert_opencv_img_to_pygame(opencv_image):
            """
            Convert OpenCV images for Pygame.

            see https://blanktar.jp/blog/2016/01/pygame-draw-opencv-image.html
            """
            opencv_image = opencv_image[:, :, ::-1]  # BGR->RGB
            shape = opencv_image.shape[1::-1]  # (height,width,Number of colors) -> (width, height)
            pygame_image = pygame.image.frombuffer(opencv_image.tobytes(), shape, 'RGB')

            return pygame_image
        h, w, _ = concat_img.shape
        print(w, h)
        target_img = convert_opencv_img_to_pygame(concat_img)
        self.screen.blit(target_img, (200, 520))
        cv2.waitKey(1)

    def set_target_images(self, images):
        super(KeyboardPlayerPyGame, self).set_target_images(images)
        self.show_target_images()

    def display_img_from_id(self, id, window_name):
        path = self.save_dir + str(id) + ".jpg"
        img = cv2.imread(path)
        cv2.imshow(window_name, img)
        cv2.waitKey(1)

    def compute_sift_features(self):
        length = len(os.listdir(self.save_dir))
        sift_descriptors = list()
        for i in range(length):
            path = str(i) + ".jpg"
            img= cv2.imread(os.path.join(self.save_dir, path))
            _, des = self.sift.detectAndCompute(img, None)
            sift_descriptors.append(des)
        return np.asarray(sift_descriptors)
    
    def get_VLAD(self, img):
        _, des = self.sift.detectAndCompute(img, None)
        pred_labels = self.codebook.predict(img)
        centroids = self.codebook.cluster_centers_
        k = self.codebook.n_clusters
        VLAD_feature = np.zeros([k, des.shape[1]])

        for i in range(k):
            if np.sum(pred_labels == i) > 0:
                VLAD_feature[i] = np.sum(des[pred_labels == i, :] - centroids[i], axis=0)
        VLAD_feature = VLAD_feature.flatten()
        VLAD_feature = np.sign(VLAD_feature) * np.sqrt(np.abs(VLAD_feature))
        VLAD_feature = VLAD_feature / np.linalg.norm(VLAD_feature)
        return VLAD_feature

    def pre_exploration(self):
        K = self.get_camera_intrinsic_matrix()
        print(f'K={K}')

    def pre_navigation(self) -> None:
        pass

    def see(self, fpv):
        if fpv is None or len(fpv.shape) < 3:
            return

        def convert_opencv_img_to_pygame(opencv_image):
            """
            Convert OpenCV images for Pygame.

            see https://blanktar.jp/blog/2016/01/pygame-draw-opencv-image.html
            """
            opencv_image = opencv_image[:, :, ::-1]  # BGR->RGB
            shape = opencv_image.shape[1::-1]  # (height,width,Number of colors) -> (width, height)
            pygame_image = pygame.image.frombuffer(opencv_image.tobytes(), shape, 'RGB')

            return pygame_image

        
        self.fpv = fpv
        if self.screen is None:
            h, w, _ = fpv.shape
            self.screen = pygame.display.set_mode((2000,1000))
            self.odom = Odometry(self.screen)

        # pygame.display.set_caption("KeyboardPlayer:fpv")
        rgb = convert_opencv_img_to_pygame(fpv)
        rgb = pygame.transform.scale(rgb, (500,375))
        self.screen.blit(rgb, (250, 100))

        pygame.display.update()
        self.odom.update(self.direction)

        if self._state:
            if self._state[1] == Phase.EXPLORATION:
                save_dir_full = os.path.join(os.getcwd(), self.save_dir)
                save_path = save_dir_full + str(self.count) + ".jpg"
            if not os.path.isdir(save_dir_full):
                os.mkdir(save_dir_full)
        cv2.imwrite(save_path, fpv)


if __name__ == "__main__":
    import logging
    logging.basicConfig(filename='vis_nav_player.log', filemode='w', level=logging.INFO,
                        format='%(asctime)s - %(levelname)s: %(message)s', datefmt='%d-%b-%y %H:%M:%S')
    import vis_nav_game as vng
    logging.info(f'player.py is using vis_nav_game {vng.core.__version__}')
    vng.play(the_player=KeyboardPlayerPyGame())
