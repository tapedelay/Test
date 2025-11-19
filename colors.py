import pygame
import numpy as np
import math
import random
import time
from pygame.locals import *
try:
    from OpenGL.GL import *
    from OpenGL.GLU import *
except ImportError:
    print("\nERROR: PyOpenGL is not installed. Please run: pip install PyOpenGL PyOpenGL_accelerate")
    exit()

# --- Configuration ---
CALC_RESOLUTION = 400
TARGET_FPS = 120
SLOW_MOTION_FACTOR = 50.0

# Physics Constants
DAMPING_FACTOR = 0.93        # Rate at which momentum decays (friction)
MOUSE_FORCE_SCALE = 0.0005   # How strongly mouse movement applies force

# 1. RESOLUTION: 1200x900
SCREEN_WIDTH, SCREEN_HEIGHT = 1200, 900 
CALC_WIDTH = 1600 
CALC_HEIGHT = 1600

# Global state flags for interaction and physics
g_is_rclick_held = False      # NEW: For continuous reorientation
g_center_velocity_x = 0.0     # NEW: Physics velocity
g_center_velocity_y = 0.0     # NEW
g_last_mouse_pos = (0, 0)     # NEW: For calculating mouse delta
g_lfo_turbo_on = False        
g_feedback_level = 0.9        # Controlled by mouse wheel

# 💡 20 VARIABLES AND THEIR RANGES
VARIABLE_PARAMS = {
    'freq_r': (0.02, 0.005, 0.05), 'freq_g': (0.015, 0.005, 0.05),
    'freq_b': (0.025, 0.01, 0.08), 'R_amp': (127.0, 60.0, 127.0),
    'G_amp': (127.0, 60.0, 127.0), 'B_amp': (127.0, 60.0, 127.0),
    'shift_x_mult': (5.0, -100.0, 100.0), 'shift_y_mult': (2.0, -100.0, 100.0), 
    'r_factor': (10.0, 5.0, 25.0), 'angle_factor': (3.0, 1.0, 8.0), # This LFO will be overridden by physics/R-Click
    'shift_center_x': (0.0, -0.2, 0.2), 'shift_center_y': (0.0, -0.2, 0.2), # These LFOs are now controlled by physics
    'time_mult_r': (1.0, 0.5, 2.0), 'time_mult_g': (1.0, 0.5, 2.0),
    'time_mult_b': (1.0, 0.5, 2.0), 'offset_r': (0.0, -10.0, 10.0),
    'offset_g': (0.0, -10.0, 10.0), 'offset_b': (0.0, -10.0, 10.0),
    'r_exponent': (1.0, 0.5, 2.0), 'blue_hue_speed': (2.0, 0.5, 4.0),
    'feedback_mix': (0.9, 0.7, 0.995) 
}
g_feedback_level = VARIABLE_PARAMS['feedback_mix'][0]

# LFO Frequencies (Global)
def generate_lfo_frequencies(params):
    frequencies = {}
    for name in params:
        # Note: We keep LFOs for shift_center_x/y/angle_factor but their values may be overridden by interaction logic
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
    
    if g_lfo_turbo_on:
        lfo_freq *= 5.0

    lfo_output = np.sin(effective_time_s * lfo_freq * 2 * np.pi)
    normalized_lfo = (lfo_output + 1.0) / 2.0
    return min_val + normalized_lfo * (max_val - min_val)


class FBO_Renderer:
    """Manages the double-buffering Frame Buffer Objects for feedback."""
    def __init__(self, width, height):
        self.width = width
        self.height = height
        if hasattr(self, 'fbos'): self.cleanup() 
        
        self.textures = glGenTextures(2)
        self.fbos = glGenFramebuffers(2)
        self.current_fbo = 0
        
        for i in range(2):
            glBindTexture(GL_TEXTURE_2D, self.textures[i])
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, None)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

            glBindFramebuffer(GL_FRAMEBUFFER, self.fbos[i])
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.textures[i], 0)
            
            if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
                 print(f"Error: FBO {i} is not complete!")

        glBindFramebuffer(GL_FRAMEBUFFER, 0)
    
    def bind_next_fbo(self):
        self.current_fbo = (self.current_fbo + 1) % 2
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbos[self.current_fbo])

    def get_input_texture(self):
        return self.textures[(self.current_fbo + 1) % 2]
    
    def cleanup(self):
        if hasattr(self, 'fbos'):
            glDeleteFramebuffers(2, self.fbos)
            glDeleteTextures(2, self.textures)
    
    def __del__(self):
        self.cleanup()


# --- GLSL Shader Source ---

VERTEX_SHADER_SOURCE = """
#version 330 core
layout(location = 0) in vec2 position;
void main() {
    gl_Position = vec4(position.x, position.y, 0.0, 1.0);
}
"""

FRAGMENT_SHADER_SOURCE = f"""
#version 330 core
out vec4 color;

uniform float u_time;
uniform vec2 u_resolution;
uniform vec2 u_mouse; 
uniform float u_reset_mix;

// LFO Uniforms
uniform float freq_r, freq_g, freq_b;
uniform float R_amp, G_amp, B_amp;
uniform float shift_x_mult, shift_y_mult;
uniform float r_factor, angle_factor; // angle_factor is manipulated by R-Click
uniform float shift_center_x, shift_center_y; // shift_center_x/y are manipulated by physics
uniform float time_mult_r, time_mult_g, time_mult_b;
uniform float offset_r, offset_g, offset_b;
uniform float r_exponent, blue_hue_speed;
uniform float feedback_mix;
uniform sampler2D u_prev_frame; 


void main() {{
    float calc_width = {CALC_WIDTH}.0;
    float calc_height = {CALC_HEIGHT}.0;
    
    float screen_scale_x = calc_width / u_resolution.x;
    float screen_scale_y = calc_height / u_resolution.y;

    float calc_x = gl_FragCoord.x * screen_scale_x;
    float calc_y = gl_FragCoord.y * screen_scale_y;
    
    // --- Mouse Interaction ---
    vec2 normalized_mouse = u_mouse / u_resolution.xy * 2.0 - 1.0; 
    
    // Core Normalized Coordinates (Non-Shifted, centered at -1.0 to 1.0)
    float nx = ((calc_x / calc_width) * 2.0 - 1.0);
    float ny = ((calc_y / calc_height) * 2.0 - 1.0);
    
    // Apply LFO/Mouse shift/Physics shift
    // The shift_center_x/y is the output of the physics engine
    nx += shift_center_x + (normalized_mouse.x * 0.15);
    ny += shift_center_y - (normalized_mouse.y * 0.15); // Invert Y for Pygame

    // Radial and Angle calculation
    float r = pow(sqrt(nx*nx + ny*ny), r_exponent);
    // angle_factor is now controlled by R-Click hold (or LFO when free)
    float angle = atan(ny, nx); 

    float blue_hue_shift = sin(u_time * blue_hue_speed) * freq_b * 100.0; 

    // --- Color Channel Calculations ---
    float red_input = freq_r * (calc_x + calc_y * time_mult_r) + shift_x_mult + offset_r;
    float Red = R_amp * (sin(red_input) + 1.0);
    
    // Green channel is where the radial pattern (r, angle) gets its structure
    float green_input = freq_g * (calc_y * r * r_factor) + angle * angle_factor + shift_y_mult + offset_g;
    float Green = G_amp * (sin(green_input) + 1.0);
    
    float blue_input = freq_b * (calc_x * 4.0 + u_time * time_mult_b) + blue_hue_shift + offset_b;
    float Blue = B_amp * (sin(blue_input) + 1.0);
    
    vec3 new_color = vec3(Red / 255.0, Green / 255.0, Blue / 255.0);

    // --- Feedback and Reset ---
    vec2 uv = gl_FragCoord.xy / u_resolution.xy; 
    vec3 prev_color = texture(u_prev_frame, uv).rgb;
    
    vec3 mixed_color = mix(new_color, prev_color, feedback_mix);
    
    // Left Click reset logic
    vec3 final_color = mix(new_color, mixed_color, u_reset_mix);
    
    color = vec4(final_color, 1.0);
}}
"""

# --- OpenGL Boilerplate Functions (Unchanged) ---
def compile_shader(source, type):
    shader = glCreateShader(type)
    glShaderSource(shader, source)
    glCompileShader(shader)
    if not glGetShaderiv(shader, GL_COMPILE_STATUS):
        raise RuntimeError(f"Shader compilation failed: \n{glGetShaderInfoLog(shader).decode('utf-8')}")
    return shader

def create_shader_program():
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
    glViewport(0, 0, width, height)
    glClearColor(0.0, 0.0, 0.0, 1.0)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(-1, 1, -1, 1, -1, 1)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

def setup_quad():
    quad_vertices = np.array([
        -1.0, -1.0, 
         1.0, -1.0, 
         1.0,  1.0, 
        -1.0,  1.0  
    ], dtype=np.float32)
    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, quad_vertices.nbytes, quad_vertices, GL_STATIC_DRAW)
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, None)
    glEnableVertexAttribArray(0)
    return vbo


# --- Main Application Loop ---

def run_real_time_codeart_gl():
    global LFO_FREQUENCIES, g_lfo_turbo_on, g_feedback_level, g_is_rclick_held
    global g_center_velocity_x, g_center_velocity_y, g_last_mouse_pos
    
    try:
        pygame.init()
        global SCREEN_WIDTH, SCREEN_HEIGHT
        
        flags = DOUBLEBUF | OPENGL | RESIZABLE
        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), flags)
        pygame.display.set_caption(f"Shader Art (Physics & Reorientation) @ {TARGET_FPS} FPS")
        
        setup_gl_context(SCREEN_WIDTH, SCREEN_HEIGHT)
        program = create_shader_program()
        vbo = setup_quad()
        fbo_manager = FBO_Renderer(SCREEN_WIDTH, SCREEN_HEIGHT)
        glUseProgram(program)

        # Get uniform locations
        uniform_locations = {name: glGetUniformLocation(program, name) for name in VARIABLE_PARAMS}
        uniform_locations['u_time'] = glGetUniformLocation(program, "u_time")
        uniform_locations['u_resolution'] = glGetUniformLocation(program, "u_resolution")
        uniform_locations['u_prev_frame'] = glGetUniformLocation(program, "u_prev_frame")
        uniform_locations['u_mouse'] = glGetUniformLocation(program, "u_mouse")
        uniform_locations['u_reset_mix'] = glGetUniformLocation(program, "u_reset_mix")

        glUniform1i(uniform_locations['u_prev_frame'], 0)

        clock = pygame.time.Clock()
        running = True
        
        mouse_x, mouse_y = SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2
        reset_mix_val = 1.0 
        
        # Initialize last mouse position for physics delta
        g_last_mouse_pos = pygame.mouse.get_pos()
        
        print(f"1/3: Interactive GPU Shader running @ {TARGET_FPS} FPS. Resolution: {SCREEN_WIDTH}x{SCREEN_HEIGHT}")
        print("2/3: Controls:")
        print("    - Left Click: Instant Visual Reset & Parameter Flip.")
        print("    - Right Click (HOLD): Continuous Reorientation (Pattern Spin).")
        print("    - Middle Click: Shuffle LFO Frequencies (New Motion Pattern).")
        print("    - Mouse Wheel: Controls Feedback/Trail Length.")
        print("    - Mouse Movement: Imparts momentum to the pattern center (Physics).")
        
        while running:
            # Time delta for physics calculations
            delta_time = clock.get_time() / 1000.0 

            # --- Event Handling ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                if event.type == pygame.VIDEORESIZE:
                    SCREEN_WIDTH, SCREEN_HEIGHT = event.size
                    pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), flags)
                    setup_gl_context(SCREEN_WIDTH, SCREEN_HEIGHT)
                    fbo_manager = FBO_Renderer(SCREEN_WIDTH, SCREEN_HEIGHT) 
                
                if event.type == pygame.MOUSEBUTTONDOWN:
                    min_fb, max_fb = VARIABLE_PARAMS['feedback_mix'][1], VARIABLE_PARAMS['feedback_mix'][2]
                    
                    if event.button == 1: # Left Click: Hard Reset & Parameter Flip
                        reset_mix_val = 0.0 
                        LFO_FREQUENCIES = generate_lfo_frequencies(VARIABLE_PARAMS)
                        print("    -> Left Click: Reset and Parameters Flipped.")
                        
                    elif event.button == 3: # Right Click: START Reorientation
                        g_is_rclick_held = True
                        print("    -> Right Click: Reorientation active (Spinning).")
                        
                    elif event.button == 2: # Middle Click: Shuffle LFO Frequencies
                        LFO_FREQUENCIES = generate_lfo_frequencies(VARIABLE_PARAMS)
                        print("    -> Middle Click: Movement Shuffled.")
                        
                    elif event.button == 4: # Mouse Wheel Up: Increase Feedback
                        g_feedback_level = min(max_fb, g_feedback_level + 0.005)
                        
                    elif event.button == 5: # Mouse Wheel Down: Decrease Feedback
                        g_feedback_level = max(min_fb, g_feedback_level - 0.005)
                
                if event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 3: # Right Click: STOP Reorientation
                        g_is_rclick_held = False
                        print("    -> Right Click: Reorientation stopped.")

            
            # --- Continuous Physics Update (Mouse & Momentum) ---
            
            current_mouse_x, current_mouse_y = pygame.mouse.get_pos()
            
            # Calculate mouse movement delta since last frame
            mouse_delta_x = current_mouse_x - g_last_mouse_pos[0]
            mouse_delta_y = current_mouse_y - g_last_mouse_pos[1]

            # Apply force to velocity based on mouse delta
            g_center_velocity_x += mouse_delta_x * MOUSE_FORCE_SCALE
            g_center_velocity_y += mouse_delta_y * MOUSE_FORCE_SCALE

            # Apply damping (friction)
            g_center_velocity_x *= DAMPING_FACTOR
            g_center_velocity_y *= DAMPING_FACTOR

            # Store current mouse position for next frame's delta calculation
            g_last_mouse_pos = (current_mouse_x, current_mouse_y)

            current_time_ms = pygame.time.get_ticks()
            effective_time_s = (current_time_ms / 1000.0) / SLOW_MOTION_FACTOR 

            # --- CPU LFO Calculation & Overrides ---
            v = {name: generate_lfo_value(name, effective_time_s) for name in VARIABLE_PARAMS}
            
            # 1. Feedback Override (Mouse Wheel)
            v['feedback_mix'] = g_feedback_level 
            
            # 2. Physics Override (Momentum applied to center shift)
            # Add velocity to the existing LFO-driven center shift base
            v['shift_center_x'] += g_center_velocity_x
            v['shift_center_y'] += g_center_velocity_y
            
            # 3. R-Click Override (Continuous Reorientation)
            if g_is_rclick_held:
                # Override the angle factor to create a sharp, continuous rotation based on time
                # Max rotation speed is around 15.0 (vs LFO max of 8.0)
                v['angle_factor'] = (math.sin(effective_time_s * 1.5) + 1.0) * 7.5 + 5.0 

            # --- 1. Render into FBO (Off-screen render) ---
            fbo_manager.bind_next_fbo()
            glViewport(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT)

            # Pass Uniforms
            glUseProgram(program)
            glUniform1f(uniform_locations['u_time'], effective_time_s)
            glUniform2f(uniform_locations['u_resolution'], float(SCREEN_WIDTH), float(SCREEN_HEIGHT))
            glUniform2f(uniform_locations['u_mouse'], float(current_mouse_x), float(current_mouse_y))
            glUniform1f(uniform_locations['u_reset_mix'], reset_mix_val)

            # Pass LFO and Physics-modified variables
            for name, value in v.items():
                glUniform1f(uniform_locations[name], value)
            
            # Bind the texture from the *previous* frame
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, fbo_manager.get_input_texture())

            # Draw the quad to update the FBO texture
            glDrawArrays(GL_QUADS, 0, 4) 
            
            # Reset the hard-reset flag
            reset_mix_val = 1.0
            
            # --- 2. Render FBO Texture to Screen (Final display) ---
            glBindFramebuffer(GL_FRAMEBUFFER, 0) 
            glViewport(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT)
            
            glDrawArrays(GL_QUADS, 0, 4)

            # Swap buffers
            pygame.display.flip()
            
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