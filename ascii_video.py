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
JAPANESE_ASCII = " ・ヽ一ノ二三八人口日月火木水金土山田中川大小上下左右出入学校雨車電話高新曜語読書買食飲見聞話"
CHINESE_ASCII = " 一丨丶丿乙二十卜人入八大天口日月田目白石竹雨山川火水木金土风云电车高新话语读写买卖食饮爱想"

class FrameProducer:
    def __init__(self, video_path, frame_queue, max_queue_size=30, start_frame=0, end_frame=None):
        self.cap = cv2.VideoCapture(video_path)
        self.frame_queue = frame_queue
        self.max_queue_size = max_queue_size
        self.running = False
        self.thread = None
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.current_frame = 0
        
        if self.start_frame > 0:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
            self.current_frame = self.start_frame
    
    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._produce_frames, daemon=True)
        self.thread.start()
        
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
        self.cap.release()
        
    def _produce_frames(self):
        while self.running:
            if self.frame_queue.qsize() >= self.max_queue_size:
                time.sleep(0.001)
                continue
                
            if self.end_frame is not None and self.current_frame >= self.end_frame:
                self.frame_queue.put(None)
                break
                
            ret, frame = self.cap.read()
            if not ret:
                self.frame_queue.put(None)
                break
                
            self.frame_queue.put(frame)
            self.current_frame += 1

class AsciiRendererGPU:
    def __init__(self, ascii_grid_width, font_size, font_path, original_video_width, original_video_height, bg_mode='adaptive', ascii_chars=None):
        if ascii_chars is None:
            ascii_chars = ASCII_CHARS
        
        self.ascii_chars = ascii_chars
        self.window = pyglet.window.Window(visible=False)
        self.ctx = moderngl.create_context()
        self.bg_mode = bg_mode
        
        self.font = ImageFont.truetype(font_path, font_size)
        ascent, descent = self.font.getmetrics()
        self.font_h = ascent + descent
        bbox_w = self.font.getbbox("X") 
        self.font_w = bbox_w[2] - bbox_w[0]
        if self.font_w <= 0:
            self.font_w = int(font_size * 0.6)
        if self.font_h <= 0:
            self.font_h = font_size
            
        video_aspect_ratio = original_video_height / original_video_width if original_video_width > 0 else 9/16
        font_aspect_ratio = self.font_w / self.font_h if self.font_h > 0 else 0.5
        
        self.ascii_grid_width = ascii_grid_width
        self.ascii_grid_height = int(self.ascii_grid_width * video_aspect_ratio * font_aspect_ratio)

        if self.ascii_grid_height <= 0:
            self.ascii_grid_height = 1
        self.num_chars = self.ascii_grid_width * self.ascii_grid_height

        atlas_img = Image.new('L', (self.font_w * len(self.ascii_chars), self.font_h), 0)
        draw = ImageDraw.Draw(atlas_img)
        for i, char in enumerate(self.ascii_chars):
            draw.text((i * self.font_w, 0), char, font=self.font, fill=255)
        self.char_atlas = self.ctx.texture(atlas_img.size, 1, atlas_img.tobytes(), dtype='f1')
        self.char_atlas.filter = (moderngl.NEAREST, moderngl.NEAREST)
        
        self.output_pixel_width = self.ascii_grid_width * self.font_w
        self.output_pixel_height = self.ascii_grid_height * self.font_h
        
        self._create_input_processing_shaders()
        self._create_main_shaders()
        self._create_blur_shaders()
        self._setup_framebuffers()
        self._setup_vertex_arrays()
        self._setup_pbo()
        
        self.frame_count = 0

    def _create_input_processing_shaders(self):
        self.preprocess_program = self.ctx.program(
            vertex_shader='''
                #version 330 core
                in vec2 position;
                out vec2 v_uv;
                void main() {
                    gl_Position = vec4(position, 0.0, 1.0);
                    v_uv = (position + 1.0) * 0.5;
                }
            ''',
            fragment_shader=f'''
                #version 330 core
                in vec2 v_uv;
                uniform sampler2D input_texture;
                uniform float brightness_factor;
                uniform float gamma;
                out vec4 f_color;
                
                void main() {{
                    vec2 flipped_uv = vec2(v_uv.x, 1.0 - v_uv.y);
                    vec3 color = texture(input_texture, flipped_uv).rgb;
                    color = pow(color * brightness_factor, vec3(1.0 / gamma));
                    float gray = dot(color, vec3(0.299, 0.587, 0.114));
                    gray = pow(gray, 0.7);
                    color = clamp(color * 1.1, 0.0, 1.0);
                    f_color = vec4(color, gray);
                }}
            '''
        )

    def _create_main_shaders(self):
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
                void main() {
                    float alpha = texture(char_atlas, v_uv).r;
                    if (alpha < 0.1) discard;
                    
                    vec3 boosted_color = v_color * brightness_boost;
                    float gray = dot(boosted_color, vec3(0.299, 0.587, 0.114));
                    boosted_color = mix(vec3(gray), boosted_color, saturation_boost);
                    
                    f_color = vec4(boosted_color * alpha, 1.0);
                }
            '''
        )
    def _create_blur_shaders(self):
        self.blur_h_program = self.ctx.program(
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
                uniform sampler2D input_texture;
                uniform vec2 texel_size;
                out vec4 f_color;
                
                void main() {
                    vec3 color = vec3(0.0);
                    float weights[5] = float[](0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);
                    
                    color += texture(input_texture, v_uv).rgb * weights[0];
                    for(int i = 1; i < 5; ++i) {
                        color += texture(input_texture, v_uv + vec2(texel_size.x * i, 0.0)).rgb * weights[i];
                        color += texture(input_texture, v_uv - vec2(texel_size.x * i, 0.0)).rgb * weights[i];
                    }
                    f_color = vec4(color * 0.4, 1.0);
                }
            '''
        )
        
        self.blur_v_program = self.ctx.program(
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
                uniform sampler2D input_texture;
                uniform vec2 texel_size;
                out vec4 f_color;
                
                void main() {
                    vec3 color = vec3(0.0);
                    float weights[5] = float[](0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);
                    
                    color += texture(input_texture, v_uv).rgb * weights[0];
                    for(int i = 1; i < 5; ++i) {
                        color += texture(input_texture, v_uv + vec2(0.0, texel_size.y * i)).rgb * weights[i];
                        color += texture(input_texture, v_uv - vec2(0.0, texel_size.y * i)).rgb * weights[i];
                    }
                    f_color = vec4(color, 1.0);
                }
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
                uniform vec3 solid_color;
                uniform int bg_mode;
                out vec4 f_color;
                void main() {
                    if (bg_mode == 0) {
                        f_color = vec4(0.0, 0.0, 0.0, 1.0);
                    } else if (bg_mode == 1) {
                        f_color = vec4(solid_color, 1.0);
                    } else {
                        f_color = vec4(texture(bg_texture, v_uv).rgb, 1.0);
                    }
                }
            '''
        )

    def _setup_framebuffers(self):
        self.main_fbo = self.ctx.framebuffer(
            color_attachments=[self.ctx.texture((self.output_pixel_width, self.output_pixel_height), 4)]
        )
        
        self.preprocess_fbo = self.ctx.framebuffer(
            color_attachments=[self.ctx.texture((self.ascii_grid_width, self.ascii_grid_height), 4)]
        )
        
        blur_w = self.output_pixel_width // 4
        blur_h = self.output_pixel_height // 4
        self.blur_fbo1 = self.ctx.framebuffer(
            color_attachments=[self.ctx.texture((blur_w, blur_h), 3)]
        )
        self.blur_fbo2 = self.ctx.framebuffer(
            color_attachments=[self.ctx.texture((blur_w, blur_h), 3)]
        )
        
        self.input_texture = self.ctx.texture((1920, 1080), 3)
        self.input_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)

    def _setup_vertex_arrays(self):
        quad_vertices = np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype=np.float32)
        quad_indices = np.array([0, 1, 2, 1, 2, 3], dtype=np.int32)
        
        self.quad_vbo = self.ctx.buffer(quad_vertices.tobytes())
        self.quad_ibo = self.ctx.buffer(quad_indices.tobytes())
        
        self.preprocess_vao = self.ctx.vertex_array(
            self.preprocess_program,
            [(self.quad_vbo, '2f', 'position')],
            self.quad_ibo
        )
        
        self.blur_h_vao = self.ctx.vertex_array(
            self.blur_h_program,
            [(self.quad_vbo, '2f', 'position')],
            self.quad_ibo
        )
        
        self.blur_v_vao = self.ctx.vertex_array(
            self.blur_v_program,
            [(self.quad_vbo, '2f', 'position')],
            self.quad_ibo
        )
        
        self.bg_vao = self.ctx.vertex_array(
            self.bg_program,
            [(self.quad_vbo, '2f', 'position')],
            self.quad_ibo
        )
        
        ascii_quad = np.array([
            -0.5, -0.5, 0.0, 0.0,
            0.5, -0.5, 1.0, 0.0,
            -0.5,  0.5, 0.0, 1.0, 
            0.5,  0.5, 1.0, 1.0 
        ], dtype=np.float32)
        self.ascii_vbo_quad = self.ctx.buffer(ascii_quad.tobytes())
        self.ascii_vbo_instance = self.ctx.buffer(reserve=self.num_chars * 6 * 4)
        self.ascii_vao = self.ctx.vertex_array(
            self.program,
            [(self.ascii_vbo_quad, '2f 2f', 'in_vert', 'in_uv'), 
             (self.ascii_vbo_instance, '2f 3f 1i /i', 'in_pos', 'in_color', 'in_char_index')],
            index_buffer=self.ctx.buffer(np.array([2, 0, 3, 1], dtype=np.int32).tobytes()),
        )
        
        x_coords = np.arange(self.ascii_grid_width, dtype=np.float32)
        y_coords = np.arange(self.ascii_grid_height, dtype=np.float32)
        xx, yy = np.meshgrid(x_coords, y_coords)
        self.instance_data = np.zeros((self.num_chars, 6), dtype=np.float32)
        self.instance_data[:, 0] = xx.flatten()
        self.instance_data[:, 1] = yy.flatten()
        
        self._setup_uniforms()

    def _setup_uniforms(self):
        self.program['char_atlas'].value = 0
        self.program['resolution'].value = (self.output_pixel_width, self.output_pixel_height)
        self.program['char_size_norm'].value = (1 / len(self.ascii_chars), 1.0)
        self.program['brightness_boost'].value = 1.8
        self.program['saturation_boost'].value = 1.6
        
        self.preprocess_program['input_texture'].value = 1
        self.preprocess_program['brightness_factor'].value = 1.0
        self.preprocess_program['gamma'].value = 1.0
        
        self.blur_h_program['input_texture'].value = 2
        self.blur_v_program['input_texture'].value = 3
        
        self.bg_program['bg_texture'].value = 4
        self.bg_program['solid_color'].value = (0.27, 0.27, 0.27)
        
        bg_mode_map = {'none': 0, 'solid': 1, 'blur': 2, 'adaptive': 2}
        self.bg_program['bg_mode'].value = bg_mode_map.get(self.bg_mode, 0)

    def _setup_pbo(self):
        self.pbo1 = self.ctx.buffer(reserve=self.output_pixel_width * self.output_pixel_height * 3)
        self.pbo2 = self.ctx.buffer(reserve=self.output_pixel_width * self.output_pixel_height * 3)
        self.current_pbo = 0
        self.pbo_ready = [False, False]

    def render_frame(self, frame):
        self.frame_count += 1
        
        if frame.shape[:2] != (self.input_texture.height, self.input_texture.width):
            self.input_texture = self.ctx.texture(frame.shape[1::-1], 3)
            self.input_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.input_texture.write(frame_rgb.tobytes())
        
        self.preprocess_fbo.use()
        self.ctx.viewport = (0, 0, self.ascii_grid_width, self.ascii_grid_height)
        self.input_texture.use(1)
        self.preprocess_vao.render()
        
        processed_data = np.frombuffer(
            self.preprocess_fbo.read(components=4, alignment=1), 
            dtype=np.uint8
        ).reshape((self.ascii_grid_height, self.ascii_grid_width, 4))

        processed_data = np.flipud(processed_data)
        
        rgb_data = processed_data[:, :, :3].astype(np.float32) / 255.0
        gray_data = processed_data[:, :, 3].astype(np.float32) / 255.0
        
        char_indices = (gray_data * (len(self.ascii_chars) - 1)).astype(np.int32)
        
        self.instance_data[:, 2:5] = rgb_data.reshape(self.num_chars, 3)
        self.instance_data[:, 5] = char_indices.flatten().view(np.float32)
        
        self.ascii_vbo_instance.write(self.instance_data.tobytes())
        
        if self.bg_mode == 'blur':
            self._render_blur_background(frame)
        elif self.bg_mode == 'adaptive':
            avg_brightness = np.mean(frame_rgb) / 255.0 * 0.6 + 0.12
            self.bg_program['solid_color'].value = (avg_brightness, avg_brightness, avg_brightness)
        
        self.main_fbo.use()
        self.ctx.viewport = (0, 0, self.output_pixel_width, self.output_pixel_height)
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        self.ctx.enable(moderngl.BLEND)
        
        self.bg_vao.render()
        
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self.char_atlas.use(0)
        self.ascii_vao.render(moderngl.TRIANGLE_STRIP, instances=self.num_chars)
        self.ctx.disable(moderngl.BLEND)
        
        current_pbo = self.pbo1 if self.current_pbo == 0 else self.pbo2
        self.main_fbo.read_into(current_pbo, components=3, alignment=1)
        
        if self.pbo_ready[1 - self.current_pbo]:
            other_pbo = self.pbo2 if self.current_pbo == 0 else self.pbo1
            image_bytes = other_pbo.read()
            img_out = np.frombuffer(image_bytes, dtype=np.uint8).reshape(
                (self.main_fbo.height, self.main_fbo.width, 3)
            )
            result = cv2.cvtColor(img_out, cv2.COLOR_RGB2BGR)
        else:
            image_bytes = self.main_fbo.read(components=3, alignment=1)
            img_out = np.frombuffer(image_bytes, dtype=np.uint8).reshape(
                (self.main_fbo.height, self.main_fbo.width, 3)
            )
            result = cv2.cvtColor(img_out, cv2.COLOR_RGB2BGR)
        
        self.pbo_ready[self.current_pbo] = True
        self.current_pbo = 1 - self.current_pbo
        
        return result
    
    def render_frames_batch(self, frames):
        results = []
        for frame in frames:
            results.append(self.render_frame(frame))
        return results

    def _render_blur_background(self, frame):
        blur_w = self.blur_fbo1.width
        blur_h = self.blur_fbo1.height
        
        self.blur_fbo1.use()
        self.ctx.viewport = (0, 0, blur_w, blur_h)
        self.input_texture.use(2)
        self.blur_h_program['texel_size'].value = (1.0 / blur_w, 1.0 / blur_h)
        self.blur_h_vao.render()
        
        self.blur_fbo2.use()
        self.blur_fbo1.color_attachments[0].use(3)
        self.blur_v_program['texel_size'].value = (1.0 / blur_w, 1.0 / blur_h)
        self.blur_v_vao.render()
        
        self.blur_fbo2.color_attachments[0].use(4)

    def release(self):
        resources = [
            'main_fbo', 'preprocess_fbo', 'blur_fbo1', 'blur_fbo2',
            'ascii_vao', 'preprocess_vao', 'blur_h_vao', 'blur_v_vao', 'bg_vao',
            'quad_vbo', 'quad_ibo', 'ascii_vbo_quad', 'ascii_vbo_instance',
            'program', 'preprocess_program', 'blur_h_program', 'blur_v_program', 'bg_program',
            'char_atlas', 'input_texture', 'pbo1', 'pbo2'
        ]
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
        if test_w <= 0:
            test_w = 8
        ideal_char_w = target_width / ascii_width
        return max(1, int((ideal_char_w / test_w) * 10))
    except IOError:
        click.echo(f"Error: Could not load font '{font_path}'. Make sure it's a valid path to a .ttf or .otf file.", err=True)
        sys.exit(1)

def process_video(input_path, output_path, width, height, grid_width, font_path, save_temp, bg_mode, use_original_res, use_batch, batch_size, start_frame, end_frame, ascii_style):
    char_sets = {
        'ascii': ASCII_CHARS,
        'japanese': JAPANESE_ASCII,
        'chinese': CHINESE_ASCII
    }
    selected_chars = char_sets[ascii_style]
    
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        click.echo(f"Error: Could not open video file {input_path}", err=True)
        return

    total_frames_original = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    original_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    if end_frame is None:
        end_frame = total_frames_original
    
    end_frame = min(end_frame, total_frames_original)
    start_frame = max(0, min(start_frame, total_frames_original - 1))
    
    if start_frame >= end_frame:
        click.echo(f"Error: Invalid frame range. Start frame ({start_frame}) must be less than end frame ({end_frame})", err=True)
        return
        
    total_frames = end_frame - start_frame
    
    if use_original_res:
        width, height = original_w, original_h
    
    font_size = calculate_font_size(width, grid_width, font_path)
    temp_video_path = output_path + ".silent.mp4"
    
    frame_queue = Queue(maxsize=30)
    producer = FrameProducer(input_path, frame_queue, start_frame=start_frame, end_frame=end_frame)
    renderer = None
    out = None

    try:
        renderer = AsciiRendererGPU(grid_width, font_size, font_path, original_w, original_h, bg_mode, selected_chars)
        out_w, out_h = (width + width % 2, height + height % 2)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_video_path, fourcc, fps, (out_w, out_h))
        
        click.echo(f"Step 1/2: Rendering frames {start_frame}-{end_frame-1} ({total_frames} total) with {bg_mode} background...")
        if use_batch:
            click.echo(f"Using batch processing with batch size: {batch_size}")
        
        producer.start()
        
        processed_frames = 0
        start_time = time.time()
        
        if use_batch:
            frames_buffer = []
            with tqdm.tqdm(total=total_frames, desc="Rendering") as pbar:
                while True:
                    frame = frame_queue.get()
                    if frame is None:
                        if frames_buffer:
                            processed_batch = renderer.render_frames_batch(frames_buffer)
                            for processed_frame in processed_batch:
                                if processed_frame.shape[:2] != (out_h, out_w):
                                    processed_frame = cv2.resize(processed_frame, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
                                out.write(processed_frame)
                                
                                processed_frames += 1
                                elapsed_time = time.time() - start_time
                                video_seconds = processed_frames / fps
                                
                                pbar.set_postfix({
                                    'Video': f'{video_seconds:.1f}s',
                                    'FPS': f'{processed_frames/elapsed_time:.1f}' if elapsed_time > 0 else '0'
                                })
                                pbar.update(1)
                        break
                        
                    frames_buffer.append(frame)
                    
                    if len(frames_buffer) >= batch_size:
                        processed_batch = renderer.render_frames_batch(frames_buffer)
                        for processed_frame in processed_batch:
                            if processed_frame.shape[:2] != (out_h, out_w):
                                processed_frame = cv2.resize(processed_frame, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
                            out.write(processed_frame)
                            
                            processed_frames += 1
                            elapsed_time = time.time() - start_time
                            video_seconds = processed_frames / fps
                            
                            pbar.set_postfix({
                                'Video': f'{video_seconds:.1f}s',
                                'FPS': f'{processed_frames/elapsed_time:.1f}' if elapsed_time > 0 else '0'
                            })
                            pbar.update(1)
                        frames_buffer = []
        else:
            with tqdm.tqdm(total=total_frames, desc="Rendering") as pbar:
                while True:
                    frame = frame_queue.get()
                    if frame is None:
                        break
                        
                    processed_frame = renderer.render_frame(frame)
                    if processed_frame.shape[:2] != (out_h, out_w):
                        processed_frame = cv2.resize(processed_frame, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
                    out.write(processed_frame)
                    
                    processed_frames += 1
                    elapsed_time = time.time() - start_time
                    video_seconds = processed_frames / fps
                    
                    pbar.set_postfix({
                        'Video': f'{video_seconds:.1f}s',
                        'FPS': f'{processed_frames/elapsed_time:.1f}' if elapsed_time > 0 else '0'
                    })
                    pbar.update(1)
        
        producer.stop()
        out.release()
        renderer.release()
        
        click.echo("\nStep 2/2: Combining video with original audio using FFmpeg...")
        
        start_time_sec = start_frame / fps
        duration_sec = total_frames / fps
        
        ffmpeg_cmd = [
            "ffmpeg", "-y", "-v", "error",
            "-ss", str(start_time_sec),
            "-t", str(duration_sec),
            "-i", input_path,
            "-i", temp_video_path,
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "128k",
            "-map", "1:v:0", "-map", "0:a:0?",
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
        if producer:
            producer.stop()
        if out and hasattr(out, 'release'):
            out.release()
        if renderer:
            renderer.release()
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
@click.option('--use-original-res', '-or', default=False, is_flag=True, help='Use original video resolution instead of custom width/height.')
@click.option('--use-batch', '-b', default=False, is_flag=True, help='Use batch processing for better GPU utilization.')
@click.option('--batch-size', default=24, help='Number of frames to process in each batch (only with --use-batch).')
@click.option('--background', '-bg', default='blur', 
              type=click.Choice(['none', 'solid', 'blur', 'adaptive']),
              help='Background mode: none (black), solid (gray), blur (blurred original), adaptive (brightness-based)')
@click.option('--start-frame', '-s', default=0, help='Start frame number (0-based index).')
@click.option('--end-frame', '-e', default=None, type=int, help='End frame number (exclusive). If not specified, processes until the end.')
@click.option('--ascii-style', default='ascii',
              type=click.Choice(['ascii', 'japanese', 'chinese']),
              help='ASCII character set to use: ascii (standard), japanese (katakana/kanji), chinese (simplified). Be sure to use font, which supports the selected character set.')
def main(input_path, output, width, height, grid_width, font_path, save_temp, use_original_res, use_batch, batch_size, background, start_frame, end_frame, ascii_style):
    click.echo()
    click.echo("--- Enhanced Video to ASCII Art Converter ---")
    click.echo(f"Input: {input_path}")
    click.echo(f"Output: {output}")
    
    cap = cv2.VideoCapture(str(input_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if use_original_res:
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        click.echo(f"Using original video resolution ({original_width}x{original_height}), Grid Width: {grid_width}")
    else:
        cap.release()
        click.echo(f"Resolution: {width}x{height}, Grid Width: {grid_width}")
    
    if end_frame is None:
        end_frame = total_frames
        
    end_frame = min(end_frame, total_frames)
    start_frame = max(0, min(start_frame, total_frames - 1))
    
    frames_to_process = end_frame - start_frame
    duration_sec = frames_to_process / fps
    
    click.echo(f"Frame range: {start_frame} to {end_frame-1} ({frames_to_process} frames, {duration_sec:.2f}s)")
    click.echo(f"Font: {font_path}")
    if use_batch:
        click.echo(f"Using batch processing with {batch_size} frames per batch.")   
    click.echo(f"ASCII Style: {ascii_style}")
    click.echo(f"Background: {background}")
    click.echo(f"Save Temp: {save_temp}")
    click.echo()
    
    start_time = time.time()
    process_video(input_path, output, width, height, grid_width, font_path, save_temp, background, use_original_res, use_batch, batch_size, start_frame, end_frame, ascii_style)
    duration = time.time() - start_time
    click.echo(f"Total operation time: {duration:.2f}s")

if __name__ == '__main__':
    main()