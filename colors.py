import pygame
import random
import math
import colorsys

# --- CONFIGURATION ---
WIDTH, HEIGHT = 1000, 1000
MAX_PARTICLES = 1500    
FRICTION = 0.98         

# --- PYGAME SETUP ---
pygame.init()
pygame.joystick.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("The Persistence of Particles (Fixed)")
clock = pygame.time.Clock()

# Controller Check
joystick = None
if pygame.joystick.get_count() > 0:
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    print(f"Connected: {joystick.get_name()}")
else:
    print("No Controller! Use Mouse.")

# --- PALETTES ---
# 0: Burning Giraffe (Orange/Gold)
# 1: Dream Sea (Teal/Blue)
# 2: Clock Melting (White/Grey/Black)
palettes = [
    [(0.05, 0.12), (0.6, 1.0)], # Hue Range, Lightness Range
    [(0.45, 0.55), (0.5, 0.9)], 
    [(0.0, 1.0),   (0.8, 1.0)], 
]
current_palette = 0
draw_crutches = False 
global_time = 0

# --- THE OBJECT ---
class MeltingObject:
    def __init__(self, x, y, angle, speed, hue, lightness, size, type="drip"):
        self.x = x
        self.y = y
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        self.hue = hue
        self.lightness = lightness
        self.base_size = size
        self.life = 1.0 
        self.type = type 
        self.decay = random.uniform(0.005, 0.01)

    def update(self, warp_x, warp_y, global_t):
        # Surreal Physics
        if self.type == "drip":
            # Sine wave gravity
            self.vx += math.sin(self.y * 0.01 + global_t) * 0.05
            self.vy += 0.05 
            # Warp
            self.vx += warp_x * 0.5
            self.vy += warp_y * 0.5
            # Friction
            self.vx *= 0.96 
            self.vy *= 0.99 
        
        elif self.type == "ant":
            # Chaos
            self.vx += random.uniform(-2, 2)
            self.vy += random.uniform(-2, 2)
            self.vx *= 0.8
            self.vy *= 0.8

        self.x += self.vx
        self.y += self.vy
        self.life -= self.decay

# --- INPUT HANDLER ---
def get_input():
    data = {
        'cursor_x': WIDTH // 2, 'cursor_y': HEIGHT // 2,
        'emit_strength': 0, 'time_melt': 0,
        'warp_x': 0, 'warp_y': 0,
        'spawn_ants': False, 'toggle_crutches': False, 'cycle_pal': False
    }

    if joystick:
        pygame.event.pump()
        
        ls_x = joystick.get_axis(0)
        ls_y = joystick.get_axis(1)
        if abs(ls_x) > 0.1 or abs(ls_y) > 0.1:
            data['cursor_x'] = (WIDTH // 2) + ls_x * (WIDTH // 2)
            data['cursor_y'] = (HEIGHT // 2) + ls_y * (HEIGHT // 2)

        data['warp_x'] = joystick.get_axis(2) 
        data['warp_y'] = joystick.get_axis(3) 

        # Triggers (RT emits, LT melts)
        data['emit_strength'] = (joystick.get_axis(5) + 1) / 2 
        data['time_melt'] = (joystick.get_axis(4) + 1) / 2 

        if joystick.get_button(0): data['spawn_ants'] = True      # A
        if joystick.get_button(3): data['toggle_crutches'] = True # Y
        if joystick.get_button(5): data['cycle_pal'] = True       # RB
    else:
        # Mouse Fallback
        mx, my = pygame.mouse.get_pos()
        data['cursor_x'] = mx
        data['cursor_y'] = my
        if pygame.mouse.get_pressed()[0]: data['emit_strength'] = 1.0
    
    return data

# --- MAIN LOOP ---
particles = []
running = True
btn_cooldown = 0

while running:
    global_time += 0.05
    inp = get_input()

    # 1. DRAW BACKGROUND (The Canvas)
    # LT controls "Time Melt" (Blur). 
    # If LT is 0, alpha is 50 (clears screen fast). If LT is 1, alpha is 5 (clears slow).
    melt_alpha = int(50 - (inp['time_melt'] * 45)) 
    melt_surface = pygame.Surface((WIDTH, HEIGHT))
    melt_surface.set_alpha(melt_alpha) 
    melt_surface.fill((20, 15, 10)) # Dark brownish canvas (not pure black)
    screen.blit(melt_surface, (0, 0))

    # 2. EVENTS
    for event in pygame.event.get():
        if event.type == pygame.QUIT: running = False

    # 3. LOGIC
    if btn_cooldown > 0: btn_cooldown -= 1
    else:
        if inp['toggle_crutches']:
            draw_crutches = not draw_crutches
            btn_cooldown = 15
        if inp['cycle_pal']:
            current_palette = (current_palette + 1) % len(palettes)
            btn_cooldown = 15
        if inp['spawn_ants']:
            for _ in range(30): # BURST
                angle = random.uniform(0, 6.28)
                speed = random.uniform(3, 10)
                # Ants have 0 Hue (Red) but very low lightness (Dark)
                p = MeltingObject(inp['cursor_x'], inp['cursor_y'], angle, speed, 0, 0.1, 3, "ant")
                particles.append(p)
            btn_cooldown = 5

    # EMITTER (Manual OR Auto-Pilot if nothing pressed)
    emit_power = inp['emit_strength']
    if emit_power < 0.05 and not inp['spawn_ants']: 
        emit_power = 0.2 # Auto-drip slightly so screen isn't empty
    
    if emit_power > 0.05:
        count = int(emit_power * 5) + 1
        for _ in range(count):
            angle = random.uniform(0, 6.28)
            speed = random.uniform(0.5, 3.0)
            
            # Color Logic
            pal = palettes[current_palette]
            hue = random.uniform(pal[0][0], pal[0][1])
            light = random.uniform(pal[1][0], pal[1][1])
            
            p = MeltingObject(inp['cursor_x'], inp['cursor_y'], angle, speed, hue, light, random.uniform(5, 15), "drip")
            particles.append(p)

    # 4. UPDATE & DRAW PARTICLES
    survivors = []
    for p in particles:
        p.update(inp['warp_x'], inp['warp_y'], global_time)
        
        if p.life > 0:
            survivors.append(p)
            
            # Color Math
            if p.type == "ant":
                color = (10, 10, 10) # Dark Grey
                # Draw Ant Halo so it's visible on dark bg
                pygame.draw.circle(screen, (100, 80, 60), (int(p.x), int(p.y)), 3)
                pygame.draw.circle(screen, color, (int(p.x), int(p.y)), 2)
            else:
                # Drips fade transparency/color
                rgb = colorsys.hls_to_rgb(p.hue, p.lightness * p.life, 1.0)
                color = (int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))

                if draw_crutches:
                    # Draw Lines
                    end_pos = (int(p.x + p.vx*10), int(p.y + p.vy*10))
                    pygame.draw.line(screen, color, (int(p.x), int(p.y)), end_pos, 2)
                else:
                    # Draw Melting Ellipses
                    stretch = 1 + (1.0 - p.life) * 4.0 
                    width = p.base_size * p.life
                    height = p.base_size * stretch
                    rect = pygame.Rect(p.x - width/2, p.y - height/2, width, height)
                    pygame.draw.ellipse(screen, color, rect)

    particles = survivors
    if len(particles) > MAX_PARTICLES:
        particles = particles[len(particles)-MAX_PARTICLES:]

    # 5. DRAW CURSOR (The "Eye")
    # Always visible so you know where you are
    eye_x, eye_y = int(inp['cursor_x']), int(inp['cursor_y'])
    # Outer Glow
    pygame.draw.circle(screen, (50, 40, 30), (eye_x, eye_y), 10, 1)
    # Inner Pupil
    pygame.draw.circle(screen, (200, 200, 200), (eye_x, eye_y), 3)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()