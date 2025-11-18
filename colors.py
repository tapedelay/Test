import pygame
import random
import math
import colorsys

# --- CONFIGURATION ---
WIDTH, HEIGHT = 1000, 1000
BG_COLOR = (0, 0, 0) 
MAX_NODES = 250      # Slightly reduced for stability in polygon mode
NODE_SPAWN_RATE = 2  

# --- PYGAME SETUP ---
pygame.init()
pygame.joystick.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Glitch Fractal Engine (Optimized)")
clock = pygame.time.Clock()
font = pygame.font.SysFont("monospace", 18)

joystick = None
if pygame.joystick.get_count() > 0:
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    print(f"Connected: {joystick.get_name()}")
else:
    print("No Controller! Using Mouse.")

# --- STATE ---
nodes = []
global_frame = 0
growth_origin_x, growth_origin_y = WIDTH // 2, HEIGHT // 2
render_mode = 0 
render_style_idx = 0 
render_styles = ["Thin Line", "Thick Line", "Dot"]

class Node:
    def __init__(self, x, y, hue_offset=0):
        self.x = x
        self.y = y
        self.vx = random.uniform(-0.5, 0.5)
        self.vy = random.uniform(-0.5, 0.5)
        self.hue = (random.random() + hue_offset) % 1.0 
        self.size = random.uniform(2, 8) 
        self.base_size = self.size 
        self.life = 1.0 
        self.decay = random.uniform(0.005, 0.015)

    def update(self, mutation_factor, tension_factor, growth_speed, origin_x, origin_y):
        # 1. Growth/Attraction
        dx_origin = origin_x - self.x
        dy_origin = origin_y - self.y
        dist_origin = math.hypot(dx_origin, dy_origin)
        
        if dist_origin > 50: 
            force_dir_x = dx_origin / dist_origin
            force_dir_y = dy_origin / dist_origin
            force_mag = (1.0 - tension_factor) * 0.5 * growth_speed
            self.vx += force_dir_x * force_mag
            self.vy += force_dir_y * force_mag

        # 2. Mutation
        self.vx += random.uniform(-mutation_factor, mutation_factor) * 0.1
        self.vy += random.uniform(-mutation_factor, mutation_factor) * 0.1

        # 3. Physics
        self.x += self.vx
        self.y += self.vy
        self.vx *= 0.95 
        self.vy *= 0.95

        # 4. Decay
        self.life -= self.decay
        self.size = max(1, int(self.base_size * self.life))

        # Boundary Warp
        if self.x < 0: self.x = WIDTH
        if self.x > WIDTH: self.x = 0
        if self.y < 0: self.y = HEIGHT
        if self.y > HEIGHT: self.y = 0


def get_input():
    data = {
        'origin_x': growth_origin_x, 'origin_y': growth_origin_y,
        'growth_accel': 0, 
        'mutation_factor': 0, 
        'tension_factor': 0,  
        'chaos_burst': False, 
        'color_glitch': 0,    
        'switch_mode': False,
        'regen_seed': False,
        'cycle_render_style': 0 
    }

    if joystick:
        pygame.event.pump()
        ls_x = joystick.get_axis(0)
        ls_y = joystick.get_axis(1)
        if abs(ls_x) > 0.1: data['origin_x'] += ls_x * 10
        if abs(ls_y) > 0.1: data['origin_y'] += ls_y * 10

        data['mutation_factor'] = joystick.get_axis(2) 
        data['tension_factor'] = joystick.get_axis(3)  
        data['growth_accel'] = (joystick.get_axis(5) + 1) / 2 
        data['color_glitch'] = (joystick.get_axis(4) + 1) / 2 

        if joystick.get_button(0): data['chaos_burst'] = True      
        if joystick.get_button(2): data['switch_mode'] = True      
        if joystick.get_button(3): data['regen_seed'] = True       
        if joystick.get_button(4): data['cycle_render_style'] = -1 
        if joystick.get_button(5): data['cycle_render_style'] = 1  
    else:
        mx, my = pygame.mouse.get_pos()
        data['origin_x'] = mx
        data['origin_y'] = my
        if pygame.mouse.get_pressed()[0]: data['growth_accel'] = 1.0
    
    data['origin_x'] = max(0, min(WIDTH, data['origin_x']))
    data['origin_y'] = max(0, min(HEIGHT, data['origin_y']))
    return data

# --- MAIN LOOP ---
running = True
btn_cooldown = 0

# Initial spawn
for _ in range(50):
    nodes.append(Node(random.randint(0, WIDTH), random.randint(0, HEIGHT)))

while running:
    screen.fill(BG_COLOR) 
    inp = get_input()
    global_frame += 1

    for event in pygame.event.get():
        if event.type == pygame.QUIT: running = False

    if btn_cooldown > 0: btn_cooldown -= 1
    else:
        if inp['switch_mode']:
            render_mode = (render_mode + 1) % 2
            btn_cooldown = 20
        if inp['regen_seed']:
            nodes.clear()
            for _ in range(50):
                nodes.append(Node(random.randint(0, WIDTH), random.randint(0, HEIGHT)))
            btn_cooldown = 15
        if inp['cycle_render_style'] != 0:
            render_style_idx = (render_style_idx + inp['cycle_render_style']) % len(render_styles)
            btn_cooldown = 15
        if inp['chaos_burst']:
            for node in nodes: 
                angle = math.atan2(node.y - growth_origin_y, node.x - growth_origin_x)
                node.vx += math.cos(angle) * 10
                node.vy += math.sin(angle) * 10
            btn_cooldown = 15

    growth_origin_x = inp['origin_x']
    growth_origin_y = inp['origin_y']

    # --- NODE SPAWNING ---
    current_spawn_rate = NODE_SPAWN_RATE + int(inp['growth_accel'] * 10)
    for _ in range(current_spawn_rate):
        if len(nodes) < MAX_NODES:
            spawn_x = growth_origin_x + random.uniform(-20, 20)
            spawn_y = growth_origin_y + random.uniform(-20, 20)
            nodes.append(Node(spawn_x, spawn_y, global_frame * 0.001))

    # --- UPDATE & RENDER ---
    survivors = []
    current_node_colors = [] 

    # 1. Update Physics first
    for node in nodes:
        node.update(abs(inp['mutation_factor']) * 5, inp['tension_factor'], 1.0 + inp['growth_accel'], growth_origin_x, growth_origin_y)
        if node.life > 0:
            survivors.append(node)
            
            # Calc color
            glitch_hue = (node.hue + inp['color_glitch'] * 0.5 + random.uniform(-inp['color_glitch']*0.1, inp['color_glitch']*0.1)) % 1.0
            lightness = 0.5 + (inp['color_glitch'] * 0.4 * math.sin(global_frame * 0.1))
            saturation = 1.0 - (inp['color_glitch'] * 0.5)
            r, g, b = colorsys.hls_to_rgb(glitch_hue, max(0.0, min(1.0, lightness)), max(0.0, min(1.0, saturation)))
            color = (int(r * 255), int(g * 255), int(b * 255))
            current_node_colors.append(color)

            if render_style_idx == 2: # Dot Mode
                pygame.draw.circle(screen, color, (int(node.x), int(node.y)), int(node.size))

    nodes = survivors

    # --- DRAW MESH ---
    if render_style_idx < 2 and len(nodes) > 1: 
        
        # Optimization: If in Polygon mode, create ONE surface for all alphas
        poly_surface = None
        if render_mode == 1:
            poly_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)

        triangles_drawn = 0
        MAX_TRIANGLES = 1000 # Hard limit to prevent freezing

        for i in range(len(nodes)):
            node1 = nodes[i]
            
            # Limit inner loop search distance to improve speed
            for j in range(i + 1, len(nodes)):
                node2 = nodes[j]
                
                # Quick check: simple distance calc
                dx = node1.x - node2.x
                dy = node1.y - node2.y
                if abs(dx) > 100 or abs(dy) > 100: continue # Optimization: Skip if far away on axis
                
                dist = math.hypot(dx, dy)
                max_dist = 150 * (1.0 - abs(inp['tension_factor'] * 0.5))
                
                if dist < max_dist:
                    line_color = (
                        int((current_node_colors[i][0] + current_node_colors[j][0]) / 2),
                        int((current_node_colors[i][1] + current_node_colors[j][1]) / 2),
                        int((current_node_colors[i][2] + current_node_colors[j][2]) / 2)
                    )
                    
                    if render_mode == 0: # Lines
                        line_thickness = 1 if render_style_idx == 0 else max(1, int(4 * node1.life))
                        pygame.draw.line(screen, line_color, (int(node1.x), int(node1.y)), (int(node2.x), int(node2.y)), line_thickness)
                    
                    elif render_mode == 1 and dist < 80 and triangles_drawn < MAX_TRIANGLES: # Polygons
                        # Find third node
                        for k in range(j + 1, len(nodes)):
                            node3 = nodes[k]
                            
                            # Distance checks
                            if abs(node2.x - node3.x) > 80 or abs(node2.y - node3.y) > 80: continue
                            dist2 = math.hypot(node2.x - node3.x, node2.y - node3.y)
                            if dist2 >= 80: continue

                            if abs(node3.x - node1.x) > 80 or abs(node3.y - node1.y) > 80: continue
                            dist3 = math.hypot(node3.x - node1.x, node3.y - node1.y)
                            
                            if dist3 < 80:
                                pts = [(int(node1.x), int(node1.y)), (int(node2.x), int(node2.y)), (int(node3.x), int(node3.y))]
                                poly_color = (line_color[0], line_color[1], line_color[2], 80) # Low alpha
                                pygame.draw.polygon(poly_surface, poly_color, pts)
                                triangles_drawn += 1

        # Blit the polygon surface once at the end
        if poly_surface:
            screen.blit(poly_surface, (0, 0))

    # UI
    pygame.draw.circle(screen, (50, 50, 50), (int(growth_origin_x), int(growth_origin_y)), 10, 1)
    
    mode_label = font.render(f"RENDER: {'POLYGONS' if render_mode == 1 else 'LINES'}", True, (200, 200, 200))
    style_label = font.render(f"STYLE: {render_styles[render_style_idx]}", True, (200, 200, 200))
    
    screen.blit(mode_label, (20, 20))
    screen.blit(style_label, (20, 45))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()