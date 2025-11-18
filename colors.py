import pygame
import math
import colorsys

# --- CONFIGURATION ---
WIDTH, HEIGHT = 1000, 1000
MAX_POINTS = 2000
BG_COLOR = (0, 0, 0) # Pure Black

# --- SETUP ---
pygame.init()
pygame.joystick.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Neon Nautilus")
clock = pygame.time.Clock()

# Controller
joystick = None
if pygame.joystick.get_count() > 0:
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    print(f"Controller: {joystick.get_name()}")

# --- STATE ---
points = [] # List of [angle, radius, speed, hue_offset]
center_x, center_y = WIDTH // 2, HEIGHT // 2
pattern_mode = 0 # 0=Simple, 1=Tri-Spiral, 2=Golden Ratio
frame_count = 0

# --- INPUTS ---
def get_input():
    data = {
        'cx': center_x, 'cy': center_y,
        'speed': 0.05, # Base auto-speed
        'warp': 1.0,
        'clear': False, 'switch': False
    }
    
    if joystick:
        pygame.event.pump()
        # Center Movement
        if abs(joystick.get_axis(0)) > 0.1: data['cx'] += joystick.get_axis(0) * 10
        if abs(joystick.get_axis(1)) > 0.1: data['cy'] += joystick.get_axis(1) * 10
        
        # Speed (RT)
        rt = (joystick.get_axis(5) + 1) / 2
        data['speed'] += rt * 0.2 # Boost speed
        
        # Warp (Right Stick X)
        if abs(joystick.get_axis(2)) > 0.1:
            data['warp'] += joystick.get_axis(2) * 0.5

        if joystick.get_button(0): data['clear'] = True      # A
        if joystick.get_button(2): data['switch'] = True     # X
    
    return data

# --- MAIN LOOP ---
running = True
btn_cooldown = 0

while running:
    screen.fill(BG_COLOR) # Keep blacks full black
    inp = get_input()
    frame_count += 1

    # Handle Events
    for event in pygame.event.get():
        if event.type == pygame.QUIT: running = False

    # Logic
    if btn_cooldown > 0: btn_cooldown -= 1
    
    if inp['clear']:
        points.clear()
    
    if inp['switch'] and btn_cooldown == 0:
        pattern_mode = (pattern_mode + 1) % 3
        points.clear()
        btn_cooldown = 20

    # Update Center
    center_x = inp['cx']
    center_y = inp['cy']

    # --- SPAWN NEW POINTS AUTOMATICALLY ---
    # This ensures the screen is never empty
    spawn_rate = 5 if inp['speed'] > 0.1 else 2
    
    for i in range(spawn_rate):
        if pattern_mode == 0: # Single Arm
            angle = frame_count * 0.1
        elif pattern_mode == 1: # Tri-Arm
            angle = frame_count * 0.1 + (i * (math.pi * 2 / 3))
        else: # Golden Ratio (Phyllotaxis)
            angle = frame_count * 137.5 * (math.pi / 180)
            
        # Start at center
        points.append([angle, 0, inp['speed'], frame_count * 0.005])

    # --- UPDATE & DRAW ---
    survivors = []
    
    for p in points:
        # p[0] = angle, p[1] = radius, p[2] = speed, p[3] = hue base
        
        # Move Outward
        p[1] += 2.0 + (p[2] * 5) # Radius growth
        
        # Rotate (Spiral effect)
        p[0] += 0.02 * inp['warp'] 

        # Cull if off screen
        if p[1] < 1200:
            survivors.append(p)
            
            # --- COLOR MATH: INCREASE WITH DISTANCE ---
            # Hue shifts as it gets further out
            dist_norm = p[1] / 600.0 # 0.0 at center, 1.0 at edge
            
            hue = (p[3] + dist_norm * 0.5) % 1.0
            lightness = 0.5
            saturation = 1.0
            
            # Convert to RGB
            r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
            color = (int(r*255), int(g*255), int(b*255))
            
            # Calculate Position
            draw_x = center_x + math.cos(p[0]) * p[1]
            draw_y = center_y + math.sin(p[0]) * p[1]
            
            # Size increases with distance
            size = max(2, int(dist_norm * 15))
            
            pygame.draw.circle(screen, color, (int(draw_x), int(draw_y)), size)

    points = survivors
    
    # Draw UI Cursor
    pygame.draw.circle(screen, (50, 50, 50), (int(center_x), int(center_y)), 5, 1)
    
    pygame.display.flip()
    clock.tick(60)

pygame.quit()