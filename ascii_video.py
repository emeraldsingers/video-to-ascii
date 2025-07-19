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

ASCII_CHARS = " .`'^\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"
FONT_PATH = r"C:\Windows\Fonts\consola.ttf"

class AsciiRendererGPU:
    def __init__(self, ascii_grid_width, font_size, original_video_width, original_video_height):
        self.window = pyglet.window.Window(visible=False)
        self.ctx = moderngl.create_context()
        self.font = ImageFont.truetype(FONT_PATH, font_size)
        ascent, descent = self.font.getmetrics()
        self.font_h = ascent + descent
        bbox_w = self.font.getbbox("X") 
        self.font_w = bbox_w[2] - bbox_w[0]
        if self.font_w <= 0: self.font_w = int(font_size * 0.6)
        if self.font_h <= 0: self.font_h = font_size
        self.ascii_grid_width = ascii_grid_width
        aspect_ratio = original_video_height / original_video_width if original_video_width > 0 else 9/16
        self.ascii_grid_height = int(self.ascii_grid_width * aspect_ratio)
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
                #version 330
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
                #version 330
                in vec3 v_color; in vec2 v_uv;
                uniform sampler2D char_atlas;
                out vec4 f_color;
                void main() {{
                    float alpha = texture(char_atlas, v_uv).r;
                    if (alpha < 0.1) discard;
                    f_color = vec4(v_color * alpha, 1.0);
                }}
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
        quad_vertices = np.array([-0.5, -0.5, 0.0, 1.0, 0.5, -0.5, 1.0, 1.0, -0.5, 0.5, 0.0, 0.0, 0.5, 0.5, 1.0, 0.0], dtype='f4')
        self.vbo_quad = self.ctx.buffer(quad_vertices)
        self.vbo_instance = self.ctx.buffer(reserve=self.num_chars * 6 * 4) 
        self.vao = self.ctx.vertex_array(
            self.program,
            [(self.vbo_quad, '2f 2f', 'in_vert', 'in_uv'), (self.vbo_instance, '2f 3f 1i /i', 'in_pos', 'in_color', 'in_char_index')],
            index_buffer=self.ctx.buffer(np.array([2, 0, 3, 1], dtype='i4')),
        )
        x_coords = np.arange(self.ascii_grid_width, dtype='f4')
        y_coords = np.arange(self.ascii_grid_height, dtype='f4')
        xx, yy = np.meshgrid(x_coords, y_coords)
        self.instance_data = np.zeros((self.num_chars, 6), dtype='f4')
        self.instance_data[:, 0] = xx.flatten()
        self.instance_data[:, 1] = yy.flatten()

    def render_frame(self, frame):
        resized_frame = cv2.resize(frame, (self.ascii_grid_width, self.ascii_grid_height), interpolation=cv2.INTER_LINEAR)
        pixels_rgb_float = (cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB).reshape(self.num_chars, 3) / 255.0).astype('f4')
        pixels_gray_float = (cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY).flatten() / 255.0).astype('f4')
        char_indices_int = (pixels_gray_float * (len(ASCII_CHARS) - 1)).astype('i4')
        self.instance_data[:, 2:5] = pixels_rgb_float
        self.instance_data[:, 5] = char_indices_int.view('f4') 
        self.vbo_instance.write(self.instance_data.tobytes())
        self.fbo.use()
        self.ctx.clear(0.0, 0.0, 0.0, 0.0)
        self.char_atlas.use(0)
        self.vao.render(moderngl.TRIANGLE_STRIP, instances=self.num_chars)
        image_bytes = self.fbo.read(components=4, alignment=1)
        img_out = np.frombuffer(image_bytes, dtype='u1').reshape((self.fbo.height, self.fbo.width, 4))
        return cv2.cvtColor(img_out, cv2.COLOR_RGBA2BGR)

    def release(self):
        resources = ['fbo', 'vao', 'vbo_quad', 'vbo_instance', 'program', 'char_atlas']
        for res_name in resources:
            if hasattr(self, res_name):
                res = getattr(self, res_name)
                if res_name == 'vao' and res and res.index_buffer:
                    res.index_buffer.release()
                if res: res.release()
        if hasattr(self, 'ctx') and self.ctx: self.ctx.release()
        if hasattr(self, 'window') and self.window: self.window.close()

def calculate_font_size(target_width, ascii_width, font_path):
    if ascii_width <= 0: raise ValueError("ASCII width must be positive.")
    try:
        test_font = ImageFont.truetype(font_path, 10)
        bbox = test_font.getbbox("X")
        test_w = bbox[2] - bbox[0]
        if test_w <= 0: test_w = 8 
        ideal_char_w = target_width / ascii_width
        return max(1, int((ideal_char_w / test_w) * 10))
    except IOError:
        click.echo(f"Error: Could not load font '{font_path}'.", err=True)
        sys.exit(1)

def process_video(input_path, output_path, width, height, grid_width):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        click.echo(f"Error: Could not open video file {input_path}", err=True)
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    original_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    font_size = calculate_font_size(width, grid_width, FONT_PATH)
    temp_video_path = output_path + ".silent.mp4"
    renderer, out = None, None

    try:
        renderer = AsciiRendererGPU(grid_width, font_size, original_w, original_h)
        out_w, out_h = (width + width % 2, height + height % 2)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_video_path, fourcc, fps, (out_w, out_h))
        
        click.echo(f"Step 1/2: Rendering {total_frames} video frames...")
        for _ in tqdm.tqdm(range(total_frames), desc="Rendering silent video"):
            ret, frame = cap.read()
            if not ret: break
            processed_frame = renderer.render_frame(frame)
            if processed_frame.shape[1] != out_w or processed_frame.shape[0] != out_h:
                final_frame = cv2.resize(processed_frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
            else:
                final_frame = processed_frame
            out.write(final_frame)
        out.release()
        renderer.release()
        
        click.echo("\nStep 2/2: Combining video with original audio using FFmpeg...")
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-i", temp_video_path, "-i", input_path,
            "-c:v", "copy", "-c:a", "aac",
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
        if 'out' in locals() and out and out.isOpened(): out.release()
        if 'renderer' in locals() and renderer: renderer.release()
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        cv2.destroyAllWindows()

@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.argument('input_path', type=click.Path(exists=True, dir_okay=False, resolve_path=True))
@click.option('--output', '-o', default='output.mp4', help='Output file path.', type=click.Path())
@click.option('--width', '-w', default=1920, help='Width of the output video in pixels.')
@click.option('--height', 'h', default=1080, help='Height of the output video in pixels.')
@click.option('--grid-width', '-g', default=300, help='Number of ASCII characters for the width (detail level).')
def main(input_path, output, width, h, grid_width):
    """
    Converts a video file to an ASCII art representation using GPU acceleration.
    The audio from the original file is copied to the final output.
    """
    click.echo("--- Video to ASCII Art Converter ---")
    click.echo(f"Input: {input_path}")
    click.echo(f"Output: {output}")
    click.echo(f"Resolution: {width}x{h}, Grid Width: {grid_width}")
    
    start_time = time.time()
    process_video(input_path, output, width, h, grid_width)
    duration = time.time() - start_time
    click.echo(f"Total operation time: {duration:.2f}s")

if __name__ == '__main__':
    main()