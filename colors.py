import math
import pygame
import random
import time
import numpy as np

# --- Configuration ---
# RENDER_RESOLUTION defines the final 600x600 display size.
RENDER_RESOLUTION = 600 
SCREEN_WIDTH = RENDER_RESOLUTION
SCREEN_HEIGHT = RENDER_RESOLUTION
TARGET_FPS = 120 # High FPS target

# 💡 MODIFIED for 8v x 4h symmetry:
# Vertical repeat unit: 600 / 8 = 75 pixels
# Horizontal repeat unit: 600 / 4 = 150 pixels
CALC_UNIT_WIDTH = 150 
CALC_UNIT_HEIGHT = 75 # <-- Changed from 300 to 75
CALC_WIDTH = 1600 # Pattern is calculated on a 1600x1600 reference grid
CALC_HEIGHT = 1600

SLOW_MOTION_FACTOR = 500.0

# Calculate the precise scale factor to map the 1600 pattern onto the 150x75 array
SCALE_FACTOR_X = CALC_WIDTH / CALC_UNIT_WIDTH
SCALE_FACTOR_Y = CALC_HEIGHT / CALC_UNIT_HEIGHT # Uses new CALC_UNIT_HEIGHT


# --- 20 VARIABLES AND THEIR RANGES (CLEANED) ---
VARIABLE_PARAMS = {
    'freq_r': (0.02, 0.005, 0.05), 'freq_g': (0.015, 0.005, 0.05),
    'freq_b': (0.025, 0.01, 0.08), 'R_amp': (127, 60, 127),
    'G_amp': (127, 60, 127), 'B_amp': (127, 60, 127),
    'shift_x_mult': (5.0, -100.0, 100.0), 'shift_y_mult': (2.0, -100.0, 100.0), 
    'r_factor': (10.0, 5.0, 25.0), 'angle_factor': (3.0, 1.0, 8.0),
    'shift_center_x': (0.0, -0.2, 0.2), 'shift_center_y': (0.0, -0.2, 0.2),
    'time_mult_r': (1.0, 0.5, 2.0), 'time_mult_g': (1.0, 0.5, 2.0),
    'time_mult_b': (1.0, 0.5, 2.0), 'offset_r': (0.0, -10.0, 10.0),
    'offset_g': (0.0, -10.0, 10.0), 'offset_b': (0.0, -10.0, 10.0),
    'r_exponent': (1.0, 0.5, 2.0), 'blue_hue_speed': (2.0, 0.5, 4.0),
}

# LFO Frequencies (Unchanged)
def generate_lfo_frequencies(params):
    frequencies = {}
    for name in params:
        if name in ['shift_center_x', 'shift_center_y']:
            frequencies[name] = random.uniform(0.0001, 0.02) / 2.0
        else:
            frequencies[name] = random.uniform(0.01, 20.0) / 2.0
    return frequencies

LFO_FREQUENCIES = generate_lfo_frequencies(VARIABLE_PARAMS)


def generate_lfo_value(variable_name, effective_time_s):
    base, min_val, max_val = VARIABLE_PARAMS[variable_name]
    lfo_freq = LFO_FREQUENCIES[variable_name]
    lfo_output = np.sin(effective_time_s * lfo_freq * 2 * np.pi)
    normalized_lfo = (lfo_output + 1.0) / 2.0
    return min_val + normalized_lfo * (max_val - min_val)

def generate_frame_numpy(v, effective_time_s):
    """
    Generates only the base unit (150x75) which is 1/32th of the final image.
    """
    # Create array matching the CALC_UNIT_HEIGHT x CALC_UNIT_WIDTH (75x150)
    y, x = np.mgrid[0:CALC_UNIT_HEIGHT, 0:CALC_UNIT_WIDTH] 
    
    # Scale coordinates by the new scale factors
    calc_x = x * SCALE_FACTOR_X
    calc_y = y * SCALE_FACTOR_Y
    
    # Normalize coordinates using the 1600 range
    nx = ((calc_x / CALC_WIDTH) * 2 - 1) + v['shift_center_x']
    ny = ((calc_y / CALC_HEIGHT) * 2 - 1) + v['shift_center_y']
    
    r = np.power(np.sqrt(nx**2 + ny**2), v['r_exponent'])
    angle = np.arctan2(ny, nx)

    shift_x = v['shift_x_mult'] 
    shift_y = v['shift_y_mult'] 
    blue_hue_shift = np.sin(effective_time_s * v['blue_hue_speed']) * v['freq_b'] * 100 

    # Calculate Color Channels (Vectorized)
    red_input = v['freq_r'] * (calc_x + calc_y * v['time_mult_r']) + shift_x + v['offset_r']
    Red = v['R_amp'] * (np.sin(red_input) + 1)

    green_input = v['freq_g'] * (calc_y * r * v['r_factor']) + angle * v['angle_factor'] + shift_y + v['offset_g']
    Green = v['G_amp'] * (np.sin(green_input) + 1)
    
    blue_input = v['freq_b'] * (calc_x * 4 + effective_time_s * v['time_mult_b']) + blue_hue_shift + v['offset_b']
    Blue = v['B_amp'] * (np.sin(blue_input) + 1)
    
    rgb_array = np.dstack((Red, Green, Blue)).astype(np.uint8)
    return rgb_array


def run_real_time_codeart():
    try:
        pygame.init()
        global SCREEN_WIDTH, SCREEN_HEIGHT
        
        flags = pygame.RESIZABLE
        # Initial screen size is 600x600
        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), flags)
        pygame.display.set_caption(f"8v x 4h Symmetry Art @ {TARGET_FPS} FPS")
        clock = pygame.time.Clock()
        
        running = True
        
        # Create the fixed-size surface for the RENDER_RESOLUTION (600x600)
        static_surface = pygame.Surface((RENDER_RESOLUTION, RENDER_RESOLUTION))

        print(f"1/3: 8v x 4h Symmetry enabled. Targeting stable {TARGET_FPS} FPS.")
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                if event.type == pygame.VIDEORESIZE:
                    SCREEN_WIDTH, SCREEN_HEIGHT = event.size
                    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), flags)

            current_time_ms = pygame.time.get_ticks()
            effective_time_s = (current_time_ms / 1000.0) / SLOW_MOTION_FACTOR 

            # --- CALCULATE ONLY THE 150x75 BASE UNIT (1/32nd of the work) ---
            v = {name: generate_lfo_value(name, effective_time_s) for name in VARIABLE_PARAMS}
            base_unit = generate_frame_numpy(v, effective_time_s)
            
            # --- NUMPY SYMMETRY COPY-PASTE (8v x 4h Stitching) ---
            
            # The base unit B is 75x150
            B = base_unit
            
            # 1. Create the base horizontal strip (1v x 4h) (75x600)
            B_flip = np.flip(B, axis=1) 
            h_strip = np.concatenate((B, B_flip, B, B_flip), axis=1)

            # 2. Create the vertical flip of the strip
            v_flip_strip = np.flip(h_strip, axis=0)
            
            # 3. Concatenate 8 alternating strips vertically (8v)
            # Pattern: H | V-flip | H | V-flip | H | V-flip | H | V-flip
            final_array = np.concatenate((
                h_strip, v_flip_strip,
                h_strip, v_flip_strip,
                h_strip, v_flip_strip,
                h_strip, v_flip_strip
            ), axis=0)

            # --- DISPLAY PHASE ---
            
            screen.fill((0, 0, 0)) 
            
            # 1. Convert the 600x600 final array to the fixed static_surface
            pygame.surfarray.blit_array(static_surface, final_array)
            
            # 2. Perform the expensive scaling operation ONCE per frame
            scaled_surface = pygame.transform.scale(static_surface, (SCREEN_WIDTH, SCREEN_HEIGHT))
            
            # 3. Blit the scaled surface
            screen.blit(scaled_surface, (0, 0))

            pygame.display.flip()
            
            # Enforce stable 120 FPS
            clock.tick(TARGET_FPS) 
            
            current_time_s = current_time_ms / 1000.0
            if current_time_s % 5 < (1 / TARGET_FPS):
                print(f"   -> Actual FPS: {clock.get_fps():.2f}")


        print("3/3: Animation closed.")
        pygame.quit()

    except ImportError:
        print("\nERROR: NumPy or Pygame is not installed.")
        print("Please run: **pip install numpy pygame**")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    run_real_time_codeart()