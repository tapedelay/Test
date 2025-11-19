import math
import pygame
import random
import time
import numpy as np

# --- Configuration ---
# Fixed resolution for calculation (400x400), ensuring CPU workload stability
CALC_RESOLUTION = 400
SCREEN_WIDTH = CALC_RESOLUTION
SCREEN_HEIGHT = CALC_RESOLUTION
TARGET_FPS = 60
PIXEL_SIZE = 1 
CALC_WIDTH = 1600 # Pattern is calculated on a 1600x1600 grid
CALC_HEIGHT = 1600

SLOW_MOTION_FACTOR = 500.0

# 💡 20 VARIABLES AND THEIR RANGES (SYNTAX CORRECTED)
VARIABLE_PARAMS = {
    'freq_r':           (0.02,   0.005,  0.05), 'freq_g': (0.015,  0.005,  0.05),
    'freq_b':           (0.025,  0.01,   0.08), 'R_amp':  (127,    60,     127),
    'G_amp':            (127,    60,     127),  'B_amp':  (127,    60,     127),
    'shift_x_mult':     (5.0,    -100.0, 100.0),'shift_y_mult':(2.0, -100.0, 100.0), 
    'r_factor':         (10.0,   5.0,    25.0), 'angle_factor':(3.0, 1.0,    8.0),
    'shift_center_x':   (0.0,    -0.2,   0.2),  'shift_center_y':(0.0, -0.2,   0.2),
    'time_mult_r':      (1.0,    0.5,    2.0),  'time_mult_g':(1.0,    0.5,    2.0),
    'time_mult_b':      (1.0,    0.5,    2.0),  'offset_r':(0.0,    -10.0,  10.0),
    'offset_g':         (0.0,    -10.0,  10.0), 'offset_b':(0.0,    -10.0,  10.0),
    'r_exponent':       (1.0,    0.5,    2.0),  'blue_hue_speed':(2.0, 0.5,    4.0),
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
    Generates the frame at the fixed CALC_RESOLUTION (400x400) using NumPy.
    """
    # Use CALC_RESOLUTION for fixed array size
    y, x = np.mgrid[0:CALC_RESOLUTION, 0:CALC_RESOLUTION] 
    
    # Scale coordinates by 4x to match the 1600x1600 pattern scale
    calc_x = x * 4
    calc_y = y * 4
    
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
        # Initial screen size is 400x400
        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), flags)
        pygame.display.set_caption(f"Resizable Stable Art @ {TARGET_FPS} FPS")
        clock = pygame.time.Clock()
        
        running = True
        
        # Create the fixed-size surface once for the NumPy result (400x400)
        static_surface = pygame.Surface((CALC_RESOLUTION, CALC_RESOLUTION))
        # Create a surface to hold the scaled output
        scaled_static_surface = static_surface.copy()

        print(f"1/3: NumPy acceleration enabled. Targeting steady {TARGET_FPS} FPS.")
        
        while running:
            # --- Event Handling (Stabilized) ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                if event.type == pygame.VIDEORESIZE:
                    # Update screen size and recreate the main screen surface
                    SCREEN_WIDTH, SCREEN_HEIGHT = event.size
                    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), flags)
                    
                    # 💡 FIX 1: When resized, perform the expensive scaling operation ONCE
                    scaled_static_surface = pygame.transform.scale(static_surface, (SCREEN_WIDTH, SCREEN_HEIGHT))

            current_time_ms = pygame.time.get_ticks()
            effective_time_s = (current_time_ms / 1000.0) / SLOW_MOTION_FACTOR 

            # --- CALCULATE FRAME using NumPy ---
            v = {name: generate_lfo_value(name, effective_time_s) for name in VARIABLE_PARAMS}
            rgb_array = generate_frame_numpy(v, effective_time_s)
            
            # --- DISPLAY PHASE (The Stable Solution) ---
            
            # 1. Clear the entire screen
            screen.fill((0, 0, 0)) 
            
            # 2. Convert NumPy array to the fixed-size static_surface (400x400)
            pygame.surfarray.blit_array(static_surface, rgb_array)
            
            # 3. 💡 FIX 2: Re-scale the newly updated static_surface to the current screen size every frame.
            # This makes the art change content AND scale, which is the final requirement.
            # This re-introduces the scaling bottleneck, but it is necessary for the visual result.
            scaled_static_surface = pygame.transform.scale(static_surface, (SCREEN_WIDTH, SCREEN_HEIGHT))
            
            # 4. Blit the scaled surface
            screen.blit(scaled_static_surface, (0, 0))

            pygame.display.flip()
            
            # Enforce stable 60 FPS
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