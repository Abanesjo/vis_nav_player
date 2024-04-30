import pygame
import math

class Odometry():
    def __init__(self, screen, SCREEN_WIDTH = 1000, SCREEN_HEIGHT = 1000):
        self.screen = screen
        self.SCREEN_WIDTH = SCREEN_WIDTH
        self.SCREEN_HEIGHT = SCREEN_HEIGHT
        self.x = 0
        self.y = 0
        self.r = 1.5
        self.theta = 0
        self.omega = math.pi * (1841/9000) * (1 - (0.84/360))
        self.fps = 15

        self.clock = pygame.time.Clock()
        self.map = pygame.surface.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.live = pygame.surface.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        self.compass = pygame.surface.Surface((60, 60), pygame.SRCALPHA)
        self.compass_icon = pygame.image.load('compass_icon.png')
        self.compass_icon = pygame.transform.scale(self.compass_icon, (50, 50))
        self.player = pygame.Rect((SCREEN_WIDTH/2, SCREEN_HEIGHT/2, 5, 5))
        self.font = pygame.font.Font(None, 36)
        self.compass_cover = pygame.Rect((0, 0, 75, 75))
    
    def draw_arrow(self, surface, x, y, angle):
        angle = angle + math.pi/2
        color = (255, 255, 255)
        length=20
        end_x = x + length * math.cos(angle)
        end_y = y - length * math.sin(angle)  # Subtract because Pygame's y-axis is inverted

        pygame.draw.line(surface, color, (x, y), (end_x, end_y), 2)

        arrow_head_angle1 = math.pi + angle - math.pi / 6  # -30 degrees in radians
        arrow_head_angle2 = math.pi + angle + math.pi / 6  # 30 degrees in radians
        arrow_head_length = length / 2  # Increase the size of the arrow head

        arrow_head_x1 = end_x + arrow_head_length * math.cos(arrow_head_angle1)
        arrow_head_y1 = end_y - arrow_head_length * math.sin(arrow_head_angle1)  # Subtract because Pygame's y-axis is inverted

        arrow_head_x2 = end_x + arrow_head_length * math.cos(arrow_head_angle2)
        arrow_head_y2 = end_y - arrow_head_length * math.sin(arrow_head_angle2)  # Subtract because Pygame's y-axis is inverted

        pygame.draw.polygon(surface, color, [(end_x, end_y), (arrow_head_x1, arrow_head_y1), (arrow_head_x2, arrow_head_y2)])
    
    def get_position(self):
        return self.x, self.y, self.theta

    def reset_position(self):
        self.x = 0
        self.y = 0
        self.theta = 0

    def update(self, direction):
        self.live.fill((0, 0, 0, 0))
        key = pygame.key.get_pressed()
        
        if self.clock.get_fps() > 0:
            self.fps = self.clock.get_fps()
        d_theta = self.omega / self.fps

        if direction=="left":
            self.theta += d_theta
        elif direction=="right":
            self.theta -= d_theta
        elif direction=="forward":
            self.x -= self.r * math.sin(self.theta)
            self.y += self.r * math.cos(self.theta)
        elif direction=="reverse":
            self.x += self.r * math.sin(self.theta)
            self.y -= self.r * math.cos(self.theta)

        self.player.x = self.x + self.SCREEN_HEIGHT/2
        self.player.y = -1 * self.y + self.SCREEN_HEIGHT/2

        rect_x = self.player.x - self.player.width // 2
        rect_y = self.player.y - self.player.height // 2
        player_rect = pygame.Rect(rect_x, rect_y, self.player.width, self.player.height)
        pygame.draw.rect(self.map, (255, 0, 0), player_rect)
        self.draw_arrow(self.live, self.player.x, self.player.y, self.theta)

        text_surface = self.font.render(f'x: {self.x:.2f}, y: {self.y:.2f}, theta: {self.theta * 180/math.pi:.2f}', True, (255, 255, 255))
        rotated_compass = pygame.transform.rotate(self.compass_icon, math.degrees(-1*self.theta))
        rect = rotated_compass.get_rect()
        rect.center = self.compass_icon.get_rect().center
        

        self.screen.blit(self.map, (self.SCREEN_WIDTH, 0))
        self.screen.blit(self.live, (self.SCREEN_WIDTH, 0))
        self.screen.blit(text_surface, (self.SCREEN_WIDTH, self.SCREEN_HEIGHT-50))
        self.compass.fill((0, 0, 255, 0))
        self.compass_cover.x = 462
        self.compass_cover.y = 15
        pygame.draw.rect(self.screen, (0, 0, 0), self.compass_cover)
        self.screen.blit(rotated_compass, (500 - rect.width // 2, 50 - rect.height // 2))
        pygame.display.update()
    
