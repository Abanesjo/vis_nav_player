import pygame
import math

pygame.init()

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 800
PI = math.pi

x = 0
y = 0
r = 2
theta = 0
omega = PI / 5
d_theta = omega /15
clock = pygame.time.Clock()

screen = pygame.display.set_mode((2 * SCREEN_WIDTH, SCREEN_HEIGHT))
map = pygame.surface.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
live = pygame.surface.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
compass_icon = pygame.image.load('compass_icon.png')
compass_icon = pygame.transform.scale(compass_icon, (50, 50))

player = pygame.Rect((SCREEN_WIDTH/2, SCREEN_HEIGHT/2, 5, 5))

def draw_arrow(surface, x, y, angle):
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

font = pygame.font.Font(None, 36)
run = True
while run:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
    
    live.fill((0, 0, 0, 0))

    key = pygame.key.get_pressed()

    if key[pygame.K_LEFT]:
        theta += d_theta
    elif key[pygame.K_RIGHT]:
        theta -= d_theta
    elif key[pygame.K_UP]:
        x -= r * math.sin(theta)
        y += r * math.cos(theta)
    elif key[pygame.K_DOWN]:
        x += r * math.sin(theta)
        y -= r * math.cos(theta)

    player.x = (x + SCREEN_WIDTH/2)
    player.y = (-1*y + SCREEN_HEIGHT/2)

    rect_x = player.x - player.width // 2
    rect_y = player.y - player.height // 2

    # Create a pygame.Rect object with the calculated top-left corner
    player_rect = pygame.Rect(rect_x, rect_y, player.width, player.height)

    # Draw the rectangle
    pygame.draw.rect(map, (255, 0, 0), player_rect)
    draw_arrow(live, player.x, player.y, theta)

    text_surface = font.render(f"({x:.2f}, {y:.2f}, {theta * 180/PI:.2f})", True, (255, 255, 255))  # Create a 
    rotated_compass = pygame.transform.rotate(compass_icon, -theta * 180/PI)
    rect = rotated_compass.get_rect()
    rect.center = compass_icon.get_rect().center


    screen.blit(map, (SCREEN_WIDTH, 0))
    screen.blit(live, (SCREEN_WIDTH, 0))
    screen.blit(text_surface, (SCREEN_WIDTH, SCREEN_HEIGHT-50))
    screen.blit(rotated_compass, (SCREEN_WIDTH +35 - rect.width // 2, 35 - rect.height // 2))
    pygame.display.update()
    clock.tick(15)

pygame.quit()