import pygame
import numpy as np
import math
import random
import time
from pygame.locals import *
# PyOpenGL is required for GPU rendering
try:
    from OpenGL.GL import *
    from OpenGL.GLU import *
except ImportError:
    print("\nERROR: PyOpenGL is not installed. Please run: pip install PyOpenGL PyOpenGL_accelerate")
    exit()

# --- Configuration ---
# Fixed resolution parameters are now mostly conceptual/for scaling constants
CALC_RESOLUTION = 400
TARGET_FPS = 60
SLOW_MOTION_FACTOR = 500.0
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600 # Initial window size
CALC_WIDTH = 1600 # Pattern is calculated on a 1600x1600 virtual grid
CALC_HEIGHT = 1600

# 💡 20 VARIABLES AND THEIR RANGES (Unchanged)
VARIABLE_PARAMS = {
    'freq_r': (0.02, 0.005, 0.05), 'freq_g': (0.015, 0.005, 0.05),
    'freq_b': (0.025, 0.01, 0.08), 'R_amp': (127.0, 60.0, 127.0),
    'G_amp': (127.0, 60.0, 127.0), 'B_amp': (127.0, 60.0, 127.0),
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
    """Calculates the current LFO value for a single variable."""
    base, min_val, max_val = VARIABLE_PARAMS[variable_name]
    lfo_freq = LFO_FREQUENCIES[variable_name]
    lfo_output = np.sin(effective_time_s * lfo_freq * 2 * np.pi)
    # Map sin output (-1 to 1) to normalized value (0 to 1)
    normalized_lfo = (lfo_output + 1.0) / 2.0
    # Map normalized value (0 to 1) to the final range (min_val to max_val)
    return min_val + normalized_lfo * (max_val - min_val)


# --- OpenGL/GLSL Shader Setup ---

# 1. Vertex Shader (Passes coordinates)
VERTEX_SHADER_SOURCE = """
#version 330 core
layout(location = 0) in vec2 position;
void main() {
    gl_Position = vec4(position.x, position.y, 0.0, 1.0);
}
"""

# 2. Fragment Shader (Replaces generate_frame_numpy)
FRAGMENT_SHADER_SOURCE = f"""
#version 330 core
out vec4 color;

// Uniforms (variables passed from the Python/CPU code)
uniform float u_time;
uniform vec2 u_resolution;
uniform float freq_r, freq_g, freq_b;
uniform float R_amp, G_amp, B_amp;
uniform float shift_x_mult, shift_y_mult;
uniform float r_factor, angle_factor;
uniform float shift_center_x, shift_center_y;
uniform float time_mult_r, time_mult_g, time_mult_b;
uniform float offset_r, offset_g, offset_b;
uniform float r_exponent, blue_hue_speed;

void main() {{
    // Global constant for pattern size
    float calc_width = {CALC_WIDTH}.0;
    float calc_height = {CALC_HEIGHT}.0;
    
    // Map gl_FragCoord (pixel coords) to the virtual 1600x1600 grid proportionally
    float screen_scale_x = calc_width / u_resolution.x;
    float screen_scale_y = calc_height / u_resolution.y;

    float calc_x = gl_FragCoord.x * screen_scale_x;
    float calc_y = gl_FragCoord.y * screen_scale_y;
    
    // Normalize coordinates using the 1600 range and center shift
    float nx = ((calc_x / calc_width) * 2.0 - 1.0) + shift_center_x;
    float ny = ((calc_y / calc_height) * 2.0 - 1.0) + shift_center_y;

    // Radial and Angle calculation
    float r = pow(sqrt(nx*nx + ny*ny), r_exponent);
    float angle = atan(ny, nx); // atan(y, x) is atan2

    // Dynamic shift
    float blue_hue_shift = sin(u_time * blue_hue_speed) * freq_b * 100.0; 

    // --- Color Channel Calculations (Translated from NumPy) ---
    // Red Channel
    float red_input = freq_r * (calc_x + calc_y * time_mult_r) + shift_x_mult + offset_r;
    float Red = R_amp * (sin(red_input) + 1.0);

    // Green Channel
    float green_input = freq_g * (calc_y * r * r_factor) + angle * angle_factor + shift_y_mult + offset_g;
    float Green = G_amp * (sin(green_input) + 1.0);
    
    // Blue Channel
    // Note: calc_x * 4 is simplified from (calc_x / 1600) * 4 * 1600, 
    // it seems the original used a coordinate four times larger than needed, keeping that logic.
    float blue_input = freq_b * (calc_x * 4.0 + u_time * time_mult_b) + blue_hue_shift + offset_b;
    float Blue = B_amp * (sin(blue_input) + 1.0);
    
    // Output color (normalized to 0.0-1.0 range, max amplitude is 254)
    vec3 final_color = vec3(Red / 255.0, Green / 255.0, Blue / 255.0);
    
    color = vec4(final_color, 1.0); // R, G, B, Alpha
}}
"""

def compile_shader(source, type):
    """Compiles a single GLSL shader."""
    shader = glCreateShader(type)
    glShaderSource(shader, source)
    glCompileShader(shader)
    
    if not glGetShaderiv(shader, GL_COMPILE_STATUS):
        # Decode error message for readability
        raise RuntimeError(f"Shader compilation failed: \n{glGetShaderInfoLog(shader).decode('utf-8')}")
    return shader

def create_shader_program():
    """Compiles and links the Vertex and Fragment shaders."""
    vertex = compile_shader(VERTEX_SHADER_SOURCE, GL_VERTEX_SHADER)
    fragment = compile_shader(FRAGMENT_SHADER_SOURCE, GL_FRAGMENT_SHADER)
    
    program = glCreateProgram()
    glAttachShader(program, vertex)
    glAttachShader(program, fragment)
    glLinkProgram(program)
    glDeleteShader(vertex)
    glDeleteShader(fragment)
    
    if not glGetProgramiv(program, GL_LINK_STATUS):
        raise RuntimeError(f"Shader linking failed: \n{glGetProgramInfoLog(program).decode('utf-8')}")
    return program

def setup_gl_context(width, height):
    """Sets up the OpenGL viewport and projection."""
    glViewport(0, 0, width, height)
    # Clear background to black
    glClearColor(0.0, 0.0, 0.0, 1.0)
    # Simple orthographic projection to map screen coordinates to (-1, 1)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(-1, 1, -1, 1, -1, 1)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

def setup_quad():
    """Creates the VBO for a full-screen quad."""
    # Vertices for a quad covering the full (-1, -1) to (1, 1) range
    quad_vertices = np.array([
        -1.0, -1.0,  # Bottom-left
         1.0, -1.0,  # Bottom-right
         1.0,  1.0,  # Top-right
        -1.0,  1.0   # Top-left
    ], dtype=np.float32)

    # Create VBO
    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, quad_vertices.nbytes, quad_vertices, GL_STATIC_DRAW)
    
    # Define how the vertex data is interpreted (location 0 is the 'position' attribute)
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, None)
    glEnableVertexAttribArray(0)
    
    return vbo

def run_real_time_codeart_gl():
    try:
        pygame.init()
        global SCREEN_WIDTH, SCREEN_HEIGHT
        
        # Initialize Pygame with OpenGL context and Double Buffering
        flags = DOUBLEBUF | OPENGL | RESIZABLE
        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), flags)
        pygame.display.set_caption(f"GPU-Accelerated Shader Art @ {TARGET_FPS} FPS")
        
        setup_gl_context(SCREEN_WIDTH, SCREEN_HEIGHT)
        program = create_shader_program()
        vbo = setup_quad()
        glUseProgram(program)

        # Get uniform locations once for efficiency
        uniform_locations = {name: glGetUniformLocation(program, name) for name in VARIABLE_PARAMS}
        uniform_locations['u_time'] = glGetUniformLocation(program, "u_time")
        uniform_locations['u_resolution'] = glGetUniformLocation(program, "u_resolution")

        clock = pygame.time.Clock()
        running = True
        
        print(f"1/3: GPU acceleration enabled. Targeting steady {TARGET_FPS} FPS.")
        
        while running:
            # --- Event Handling ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                if event.type == pygame.VIDEORESIZE:
                    # Update screen size, recreate Pygame window, and reset GL viewport
                    SCREEN_WIDTH, SCREEN_HEIGHT = event.size
                    pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), flags)
                    setup_gl_context(SCREEN_WIDTH, SCREEN_HEIGHT)
            
            current_time_ms = pygame.time.get_ticks()
            effective_time_s = (current_time_ms / 1000.0) / SLOW_MOTION_FACTOR 

            # --- CPU LFO Calculation ---
            # Calculate all 20 LFO values on the CPU
            v = {name: generate_lfo_value(name, effective_time_s) for name in VARIABLE_PARAMS}
            
            # --- Pass Uniforms to GPU ---
            glUseProgram(program)
            
            # Pass Time and Resolution
            glUniform1f(uniform_locations['u_time'], effective_time_s)
            glUniform2f(uniform_locations['u_resolution'], float(SCREEN_WIDTH), float(SCREEN_HEIGHT))
            
            # Pass all 20 LFO variables
            for name, value in v.items():
                glUniform1f(uniform_locations[name], value)

            # --- RENDER PHASE ---
            glClear(GL_COLOR_BUFFER_BIT) # Clear the screen
            
            # Draw the quad: This triggers the Fragment Shader for every pixel
            glDrawArrays(GL_QUADS, 0, 4) 
            
            # Swap buffers to display the new frame
            pygame.display.flip()
            
            # Enforce stable 60 FPS
            clock.tick(TARGET_FPS) 
            
            current_time_s = current_time_ms / 1000.0
            if current_time_s % 5 < (1 / TARGET_FPS):
                print(f"  -> Actual FPS: {clock.get_fps():.2f}")

        print("3/3: Animation closed.")
        # Cleanup
        glDeleteProgram(program)
        glDeleteBuffers(1, [vbo])
        pygame.quit()

    except Exception as e:
        print(f"An unexpected error occurred during execution: {e}")
        pygame.quit()

if __name__ == "__main__":
    run_real_time_codeart_gl()