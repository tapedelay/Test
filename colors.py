import pygame
import random
import math
import colorsys

# --- CONFIGURATION ---
WIDTH, HEIGHT = 1000, 1000
MAX_PARTICLES = 1500    # Keep this reasonable for CPU rendering
FRICTION = 0.96         # Particles slow down over time
GRAVITY_STRENGTH = 0.8

# --- PYGAME SETUP ---
pygame.init()
pygame.joystick.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Neon Kaleidoscope Synth")
clock = pygame.time.Clock()

# Controller Check
joystick = None
if pygame.joystick.get_count() > 0:
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    print(f"Connected: {joystick.get_name()}")

# --- STATE MANAGEMENT ---
particles = []
palettes = [
    (0.0, 0.1),   # Red/Orange
    (0.3, 0.4),   # Green/Teal
    (0.5, 0.7),   # Blue/Purple
    (0.8, 0.95),  # Pink/Red
    (0.0, 1.0)    # Full Rainbow
]
current_palette = 4
symmetry_modes = [1, 2, 4, 6, 8, 12] # Number of mirrors
symmetry_index = 4 # Start at 8x symmetry
draw_lines = False # Toggle between dots and lines
global_rotation = 0.0 # Right stick rotates the world

# --- HELPER CLASSES ---
class Particle:
    def __init__(self, x, y, angle, speed, color, size):
        self.x = x
        self.y = y
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        self.color = color
        self.size = size
        self.life = 1.0 # 1.0 = 100% life, 0.0 = Dead
        self.decay = random.uniform(0.01, 0.03) # How fast it dies

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vx *= FRICTION
        self.vy *= FRICTION
        self.life -= self.decay

# --- INPUT HANDLER ---
def get_input():
    data = {
        'cursor_x': WIDTH // 2, 'cursor_y': HEIGHT // 2,
        'emit_strength': 0, 'suck_strength': 0,
        'rot_speed': 0,
        'explode': False, 'cycle_sym': False, 
        'toggle_lines': False, 'cycle_pal': False
    }

    if joystick:
        pygame.event.pump()
        
        # 1. CURSOR (Left Stick)
        # Map stick (-1 to 1) to screen coordinates
        ls_x = joystick.get_axis(0)
        ls_y = joystick.get_axis(1)
        if abs(ls_x) > 0.1 or abs(ls_y) > 0.1:
            data['cursor_x'] = (WIDTH // 2) + ls_x * (WIDTH // 2)
            data['cursor_y'] = (HEIGHT // 2) + ls_y * (HEIGHT // 2)

        # 2. WORLD ROTATION (Right Stick X)
        rs_x = joystick.get_axis(2)
        if abs(rs_x) > 0.1:
            data['rot_speed'] = rs_x * 0.05

        # 3. TRIGGERS
        # Convert (-1 to 1) -> (0 to 1)
        data['emit_strength'] = (joystick.get_axis(5) + 1) / 2 # RT
        data['suck_strength'] = (joystick.get_axis(4) + 1) / 2 # LT

        # 4. BUTTONS
        if joystick.get_button(0): data['explode'] = True      # A
        if joystick.get_button(2): data['cycle_sym'] = True    # X
        if joystick.get_button(3): data['toggle_lines'] = True # Y
        if joystick.get_button(5): data['cycle_pal'] = True    # RB
    
    # MOUSE FALLBACK (If controller is disconnected)
    else:
        mx, my = pygame.mouse.get_pos()
        buttons = pygame.mouse.get_pressed()
        data['cursor_x'] = mx
        data['cursor_y'] = my
        if buttons[0]: data['emit_strength'] = 1.0
        if buttons[2]: data['suck_strength'] = 1.0

    return data

# --- RENDERER (THE MAGIC) ---
def draw_symmetric(surface, x, y, size, color, symmetry, rot_offset):
    # Center of screen
    cx, cy = WIDTH // 2, HEIGHT // 2
    
    # Calculate relative position from center
    rel_x = x - cx
    rel_y = y - cy
    
    # Convert to polar coordinates (radius, angle)
    radius = math.hypot(rel_x, rel_y)
    base_angle = math.atan2(rel_y, rel_x)
    
    # Draw N times rotated
    angle_step = (math.pi * 2) / symmetry
    
    for i in range(symmetry):
        # Current rotation angle
        theta = base_angle + (i * angle_step) + rot_offset
        
        # Convert back to Cartesian
        draw_x = cx + radius * math.cos(theta)
        draw_y = cy + radius * math.sin(theta)
        
        # Draw
        # If size is small, a pixel is faster than a circle
        if size < 2:
            surface.set_at((int(draw_x), int(draw_y)), color)
        else:
            pygame.draw.circle(surface, color, (int(draw_x), int(draw_y)), int(size))

# --- MAIN LOOP ---
running = True
btn_cooldown = 0

while running:
    # 1. Setup Frame
    # Instead of clearing to black, we draw a semi-transparent black rectangle
    # This creates "Trails" or "Motion Blur"
    fade_surface = pygame.Surface((WIDTH, HEIGHT))
    fade_surface.set_alpha(40) # 0-255 (Lower = Longer trails)
    fade_surface.fill((0, 0, 0))
    screen.blit(fade_surface, (0, 0))

    # 2. Events
    for event in pygame.event.get():
        if event.type == pygame.QUIT: running = False

    # 3. Logic
    inp = get_input()
    
    # Handle Global Rotation
    global_rotation += inp['rot_speed']

    # Handle Button Toggles (Cooldown prevents rapid flickering)
    if btn_cooldown > 0: btn_cooldown -= 1
    else:
        if inp['cycle_sym']:
            symmetry_index = (symmetry_index + 1) % len(symmetry_modes)
            symmetry = symmetry_modes[symmetry_index]
            print(f"Symmetry: {symmetry}x")
            btn_cooldown = 15
        if inp['toggle_lines']:
            draw_lines = not draw_lines
            btn_cooldown = 15
        if inp['cycle_pal']:
            current_palette = (current_palette + 1) % len(palettes)
            btn_cooldown = 15
        if inp['explode']:
            # Spawn a ring explosion
            for i in range(50):
                angle = random.uniform(0, 6.28)
                speed = random.uniform(5, 15)
                hue = random.uniform(*palettes[current_palette])
                rgb = colorsys.hls_to_rgb(hue, 0.6, 1.0)
                col = (int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
                p = Particle(WIDTH//2, HEIGHT//2, angle, speed, col, random.uniform(3, 8))
                particles.append(p)
            btn_cooldown = 10

    # 4. Particle Spawning (Right Trigger)
    if inp['emit_strength'] > 0.05:
        # Spawn rate depends on how hard trigger is pressed
        count = int(inp['emit_strength'] * 10) 
        for _ in range(count):
            # Random spread
            angle = random.uniform(0, 6.28)
            speed = random.uniform(1, 5) + (inp['emit_strength'] * 5)
            
            # Color logic
            h_min, h_max = palettes[current_palette]
            hue = random.uniform(h_min, h_max)
            rgb = colorsys.hls_to_rgb(hue, 0.5, 1.0)
            col = (int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
            
            p = Particle(inp['cursor_x'], inp['cursor_y'], angle, speed, col, random.uniform(2, 5))
            particles.append(p)

    # 5. Update Particles
    survivors = []
    cx, cy = WIDTH // 2, HEIGHT // 2 # Center
    
    for p in particles:
        # Apply Physics
        p.update()
        
        # LEFT TRIGGER: Gravity Well / Black Hole
        if inp['suck_strength'] > 0.05:
            dx = inp['cursor_x'] - p.x
            dy = inp['cursor_y'] - p.y
            dist = math.hypot(dx, dy)
            if dist > 10: # Don't divide by zero
                # Force is inversely proportional to distance
                force = (inp['suck_strength'] * 1000) / (dist * dist)
                force = min(force, 2.0) # Cap the force
                p.vx += (dx / dist) * force
                p.vy += (dy / dist) * force

        # Keep living particles
        if p.life > 0:
            survivors.append(p)
            
            # Calculate brightness based on life
            # Fade out color
            faded_color = (
                int(p.color[0] * p.life),
                int(p.color[1] * p.life),
                int(p.color[2] * p.life)
            )
            
            # DRAW IT
            sym_count = symmetry_modes[symmetry_index]
            draw_symmetric(screen, p.x, p.y, p.size * p.life, faded_color, sym_count, global_rotation)

    particles = survivors

    # Limit particle count (Delete oldest if too many)
    if len(particles) > MAX_PARTICLES:
        particles = particles[len(particles)-MAX_PARTICLES:]

    # 6. UI / HUD (Minimal)
    # Show cursor only if not emitting (so you can see where you are)
    if inp['emit_strength'] < 0.1:
        pygame.draw.circle(screen, (50, 50, 50), (int(inp['cursor_x']), int(inp['cursor_y'])), 5, 1)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()