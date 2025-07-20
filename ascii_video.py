# -*- coding: utf-8 -*-
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import tqdm
import moderngl
import pyglet.window
import sys
import subprocess
import time
import click
from concurrent.futures import ThreadPoolExecutor
import threading
from queue import Queue

ASCII_CHARS = " .`'^\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$░▒▓█"

class AsciiRendererGPU:
    def __init__(self, ascii_grid_width, font_size, font_path, original_video_width, original_video_height, bg_mode='adaptive'):
        self.window = pyglet.window.Window(visible=False)
        self.ctx = moderngl.create_context()
        self.bg_mode = bg_mode
        
        self.font = ImageFont.truetype(font_path, font_size)
        ascent, descent = self.font.getmetrics()
        self.font_h = ascent + descent
        bbox_w = self.font.getbbox("X") 
        self.font_w = bbox_w[2] - bbox_w[0]
        if self.font_w <= 0: self.font_w = int(font_size * 0.6)
        if self.font_h <= 0: self.font_h = font_size
        video_aspect_ratio = original_video_height / original_video_width if original_video_width > 0 else 9/16
        font_aspect_ratio = self.font_w / self.font_h if self.font_h > 0 else 0.5
        
        self.ascii_grid_width = ascii_grid_width
        self.ascii_grid_height = int(self.ascii_grid_width * video_aspect_ratio * font_aspect_ratio)

        if self.ascii_grid_height <= 0: self.ascii_grid_height = 1
        self.num_chars = self.ascii_grid_width * self.ascii_grid_height

        atlas_img = Image.new('L', (self.font_w * len(ASCII_CHARS), self.font_h), 0)
        draw = ImageDraw.Draw(atlas_img)
        for i, char in enumerate(ASCII_CHARS):
            draw.text((i * self.font_w, 0), char, font=self.font, fill=255)
        self.char_atlas = self.ctx.texture(atlas_img.size, 1, atlas_img.tobytes(), dtype='f1')
        self.char_atlas.filter = (moderngl.NEAREST, moderngl.NEAREST)
        
        self.program = self.ctx.program(
            vertex_shader=f'''
                #version 330 core
                in vec2 in_vert; in vec2 in_uv; in vec2 in_pos; in vec3 in_color; in int in_char_index;
                uniform vec2 resolution; uniform vec2 char_size_norm;
                out vec3 v_color; out vec2 v_uv;
                void main() {{
                    v_uv = vec2(in_uv.x * char_size_norm.x + in_char_index * char_size_norm.x, in_uv.y);
                    v_color = in_color;
                    vec2 char_pixel_size = vec2({self.font_w}.0, {self.font_h}.0);
                    vec2 pos = (in_pos + in_vert + 0.5) * char_pixel_size * 2.0 / resolution - 1.0;
                    gl_Position = vec4(pos.x, pos.y, 0.0, 1.0);
                }}
            ''',
            fragment_shader='''
                #version 330 core
                in vec3 v_color; in vec2 v_uv;
                uniform sampler2D char_atlas;
                uniform float brightness_boost;
                uniform float saturation_boost;
                out vec4 f_color;
                void main() {{
                    float alpha = texture(char_atlas, v_uv).r;
                    if (alpha < 0.1) discard;
                    
                    vec3 boosted_color = v_color * brightness_boost;
                    float gray = dot(boosted_color, vec3(0.299, 0.587, 0.114));
                    boosted_color = mix(vec3(gray), boosted_color, saturation_boost);
                    
                    f_color = vec4(boosted_color * alpha, 1.0);
                }}
            '''
        )
        
        self.bg_program = self.ctx.program(
            vertex_shader='''
                #version 330 core
                in vec2 position;
                out vec2 v_uv;
                void main() {
                    gl_Position = vec4(position, 0.0, 1.0);
                    v_uv = (position + 1.0) * 0.5;
                }
            ''',
            fragment_shader='''
                #version 330 core
                in vec2 v_uv;
                uniform sampler2D bg_texture;
                uniform float bg_alpha;
                out vec4 f_color;
                void main() {
                    vec3 bg_color = texture(bg_texture, v_uv).rgb;
                    f_color = vec4(bg_color * bg_alpha, 1.0);
                }
            '''
        )
        
        self.output_pixel_width = self.ascii_grid_width * self.font_w
        self.output_pixel_height = self.ascii_grid_height * self.font_h
        self.fbo = self.ctx.framebuffer(
            color_attachments=[self.ctx.texture((self.output_pixel_width, self.output_pixel_height), 4)]
        )
        
        self.program['char_atlas'].value = 0
        self.program['resolution'].value = (self.output_pixel_width, self.output_pixel_height)
        self.program['char_size_norm'].value = (1 / len(ASCII_CHARS), 1.0)
        self.program['brightness_boost'].value = 1.8
        self.program['saturation_boost'].value = 1.6
        
        self.bg_program['bg_texture'].value = 1
        self.bg_program['bg_alpha'].value = 0.4 
        
        quad_vertices = np.array([-0.5, -0.5, 0.0, 1.0, 0.5, -0.5, 1.0, 1.0, -0.5, 0.5, 0.0, 0.0, 0.5, 0.5, 1.0, 0.0], dtype=np.float32)
        self.vbo_quad = self.ctx.buffer(quad_vertices.tobytes())
        self.vbo_instance = self.ctx.buffer(reserve=self.num_chars * 6 * 4) 
        self.vao = self.ctx.vertex_array(
            self.program,
            [(self.vbo_quad, '2f 2f', 'in_vert', 'in_uv'), (self.vbo_instance, '2f 3f 1i /i', 'in_pos', 'in_color', 'in_char_index')],
            index_buffer=self.ctx.buffer(np.array([2, 0, 3, 1], dtype=np.int32).tobytes()),
        )
        
        bg_quad = np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype=np.float32)
        self.bg_vbo = self.ctx.buffer(bg_quad.tobytes())
        self.bg_vao = self.ctx.vertex_array(
            self.bg_program,
            [(self.bg_vbo, '2f', 'position')],
            index_buffer=self.ctx.buffer(np.array([0, 1, 2, 1, 2, 3], dtype=np.int32).tobytes())
        )
        
        x_coords = np.arange(self.ascii_grid_width, dtype=np.float32)
        y_coords = np.arange(self.ascii_grid_height, dtype=np.float32)
        xx, yy = np.meshgrid(x_coords, y_coords)
        self.instance_data = np.zeros((self.num_chars, 6), dtype=np.float32)
        self.instance_data[:, 0] = xx.flatten()
        self.instance_data[:, 1] = yy.flatten()
        self.bg_texture = self.ctx.texture((self.output_pixel_width, self.output_pixel_height), 3)
        self.bg_texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
        
        self.blur_cache = None
        self.blur_interval = 5
        self.frame_count = 0

    def create_background_fast(self, frame):
        if self.bg_mode == 'none':
            return np.zeros((self.output_pixel_height, self.output_pixel_width, 3), dtype=np.uint8)
        elif self.bg_mode == 'blur':
            if self.blur_cache is None or self.frame_count % self.blur_interval == 0:
                small_frame = cv2.resize(frame, (self.output_pixel_width // 4, self.output_pixel_height // 4))
                blurred = cv2.GaussianBlur(small_frame, (15, 15), 0)
                self.blur_cache = cv2.resize(blurred, (self.output_pixel_width, self.output_pixel_height))
            bg_data = (cv2.cvtColor(self.blur_cache, cv2.COLOR_BGR2RGB) * 0.4).astype(np.uint8)
        elif self.bg_mode == 'adaptive':
            avg_brightness = int(np.mean(frame) * 0.6 + 30)
            bg_data = np.full((self.output_pixel_height, self.output_pixel_width, 3), avg_brightness, dtype=np.uint8)
        else:
            bg_data = np.full((self.output_pixel_height, self.output_pixel_width, 3), 70, dtype=np.uint8)
        return bg_data

    def render_frame(self, frame):
        self.frame_count += 1
        
        resized_frame = cv2.resize(frame, (self.ascii_grid_width, self.ascii_grid_height), interpolation=cv2.INTER_NEAREST)
        
        rgb_data = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        gray_data = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        
        gray_data = np.power(gray_data, 0.7)
        char_indices = (gray_data * (len(ASCII_CHARS) - 1)).astype(np.int32)
        
        rgb_data = np.clip(rgb_data * 1.1, 0, 1)
        
        self.instance_data[:, 2:5] = rgb_data.reshape(self.num_chars, 3)
        self.instance_data[:, 5] = char_indices.flatten().view(np.float32)
        
        self.vbo_instance.write(self.instance_data.tobytes())
        
        bg_data = self.create_background_fast(frame)
        self.bg_texture.write(bg_data.tobytes())
        
        self.fbo.use()
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)  
        self.ctx.enable(moderngl.BLEND)
        
        if self.bg_mode != 'none':
            self.ctx.blend_func = moderngl.ONE, moderngl.ZERO 
            self.bg_texture.use(1)
            self.bg_vao.render(moderngl.TRIANGLES)
            self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        
        self.char_atlas.use(0)
        self.vao.render(moderngl.TRIANGLE_STRIP, instances=self.num_chars)
        self.ctx.disable(moderngl.BLEND)
        
        image_bytes = self.fbo.read(components=3, alignment=1)
        img_out = np.frombuffer(image_bytes, dtype=np.uint8).reshape((self.fbo.height, self.fbo.width, 3))
        return cv2.cvtColor(img_out, cv2.COLOR_RGB2BGR)
    
    def render_frames_batch(self, frames):
        results = []
        for frame in frames:
            results.append(self.render_frame(frame))
        return results

    def release(self):
        resources = ['fbo', 'vao', 'bg_vao', 'vbo_quad', 'bg_vbo', 'vbo_instance', 'program', 'bg_program', 'char_atlas', 'bg_texture']
        for res_name in resources:
            if hasattr(self, res_name):
                res = getattr(self, res_name)
                if hasattr(res, 'release') and callable(res.release):
                    res.release()
        if hasattr(self, 'ctx') and self.ctx: 
            self.ctx.release()
        if hasattr(self, 'window') and self.window: 
            self.window.close()

def calculate_font_size(target_width, ascii_width, font_path):
    if ascii_width <= 0: 
        raise ValueError("ASCII width must be positive.")
    try:
        test_font = ImageFont.truetype(font_path, 10)
        bbox = test_font.getbbox("X")
        test_w = bbox[2] - bbox[0]
        if test_w <= 0: test_w = 8 
        ideal_char_w = target_width / ascii_width
        return max(1, int((ideal_char_w / test_w) * 10))
    except IOError:
        click.echo(f"Error: Could not load font '{font_path}'. Make sure it's a valid path to a .ttf or .otf file.", err=True)
        sys.exit(1)

def process_frames_batch(frames, renderer):
    results = []
    for frame in frames:
        results.append(renderer.render_frame(frame))
    return results

def process_video(input_path, output_path, width, height, grid_width, font_path, save_temp, bg_mode, use_batch_size, batch_size):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        click.echo(f"Error: Could not open video file {input_path}", err=True)
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    original_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    font_size = calculate_font_size(width, grid_width, font_path)
    temp_video_path = output_path + ".silent.mp4"
    renderer, out = None, None

    try:
        renderer = AsciiRendererGPU(grid_width, font_size, font_path, original_w, original_h, bg_mode)
        out_w, out_h = (width + width % 2, height + height % 2)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_video_path, fourcc, fps, (out_w, out_h))
        
        click.echo(f"Step 1/2: Rendering {total_frames} video frames with {bg_mode} background...")
        if use_batch_size:
            click.echo(f"Using batch processing with batch size: {batch_size}")
        
        if use_batch_size:
            frames_buffer = []
            with tqdm.tqdm(total=total_frames, desc="Rendering") as pbar:
                for _ in range(total_frames):
                    ret, frame = cap.read()
                    if not ret: 
                        break
                        
                    frames_buffer.append(frame)
                    
                    if len(frames_buffer) >= batch_size:
                        processed_frames = renderer.render_frames_batch(frames_buffer)
                        for processed_frame in processed_frames:
                            if processed_frame.shape[:2] != (out_h, out_w):
                                processed_frame = cv2.resize(processed_frame, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
                            out.write(processed_frame)
                        frames_buffer = []
                        pbar.update(batch_size)
                
                if frames_buffer:
                    processed_frames = renderer.render_frames_batch(frames_buffer)
                    for processed_frame in processed_frames:
                        if processed_frame.shape[:2] != (out_h, out_w):
                            processed_frame = cv2.resize(processed_frame, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
                        out.write(processed_frame)
                        pbar.update(1)
        else:
            with tqdm.tqdm(total=total_frames, desc="Rendering") as pbar:
                for _ in range(total_frames):
                    ret, frame = cap.read()
                    if not ret: 
                        break
                        
                    processed_frame = renderer.render_frame(frame)
                    if processed_frame.shape[:2] != (out_h, out_w):
                        processed_frame = cv2.resize(processed_frame, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
                    out.write(processed_frame)
                    pbar.update(1)
        
        out.release()
        renderer.release()
        
        click.echo("\nStep 2/2: Combining video with original audio using FFmpeg...")
        ffmpeg_cmd = [
            "ffmpeg", "-y", "-v", "error",
            "-i", temp_video_path, "-i", input_path,
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "20",
            "-c:a", "aac", "-b:a", "128k",
            "-map", "0:v:0", "-map", "1:a:0?",
            "-shortest", output_path
        ]
        subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True)
        click.secho(f"\nVideo successfully saved to: {output_path}", fg='green')

    except FileNotFoundError:
        click.echo("\nERROR: FFmpeg not found. Please ensure FFmpeg is installed and in your system's PATH.", err=True)
    except subprocess.CalledProcessError as e:
        click.echo(f"\nERROR: FFmpeg execution failed (code {e.returncode}):", err=True)
        click.echo(e.stderr, err=True)
    except Exception as e:
        click.echo(f"\nAn unexpected error occurred: {e}", err=True)
    finally:
        if cap: cap.release()
        if 'out' in locals() and out and hasattr(out, 'release'): out.release()
        if 'renderer' in locals() and renderer: renderer.release()
        if not save_temp and os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        cv2.destroyAllWindows()

@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.argument('input_path', type=click.Path(exists=True, dir_okay=False, resolve_path=True))
@click.option('--output', '-o', default='output.mp4', help='Output file path.', type=click.Path())
@click.option('--width', '-W', default=1920, help='Width of the output video in pixels.')
@click.option('--height', '-H', default=1080, help='Height of the output video in pixels.')
@click.option('--grid-width', '-g', default=300, help='Number of ASCII characters for the width (detail level).')
@click.option('--font', 'font_path', default=r"C:\Windows\Fonts\consola.ttf", help='Path to a MONOSPACED font file (.ttf, .otf).', type=click.Path(exists=True, dir_okay=False))
@click.option('--save-temp', default=False, is_flag=True, help='Save the temporary silent video file.')
@click.option('--use-batch', '-b', default=False, is_flag=True, help='Use batch processing for better GPU utilization.')
@click.option('--batch-size', default=64, help='Number of frames to process in each batch (only with --use-batch).')
@click.option('--background', '-bg', default='blur', 
              type=click.Choice(['none', 'solid', 'blur', 'adaptive']),
              help='Background mode: none (black), solid (gray), blur (blurred original), adaptive (brightness-based)')
def main(input_path, output, width, height, grid_width, font_path, save_temp, use_batch, batch_size, background):
    """
    Converts a video file to an ASCII art representation using GPU acceleration.
    The audio from the original file is copied to the final output.
    
    Background modes:
    - none: Black background (original behavior)
    - solid: Light gray background (fast, but ugly)
    - blur: Blurred and darkened version of original frame (recommended, but really slow)
    - adaptive: Background brightness adapts to frame brightness (fast enough, but shitty)
    """
    click.echo("--- Enhanced Video to ASCII Art Converter ---")
    click.echo(f"Input: {input_path}")
    click.echo(f"Output: {output}")
    click.echo(f"Resolution: {width}x{height}, Grid Width: {grid_width}")
    click.echo(f"Font: {font_path}")
    click.echo(f"Background: {background}")
    click.echo(f"Batch processing: {use_batch}")
    if use_batch:
        click.echo(f"Batch size: {batch_size}")
    click.echo(f"Save Temp: {save_temp}")
    click.echo()
    
    start_time = time.time()
    process_video(input_path, output, width, height, grid_width, font_path, save_temp, background, use_batch, batch_size)
    duration = time.time() - start_time
    click.echo(f"Total operation time: {duration:.2f}s")

if __name__ == '__main__':
    main()