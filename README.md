# Video to ASCII Art Converter (GPU Accelerated)

This script converts video files into a stylized ASCII art representation, leveraging the GPU for high-performance rendering. The program preserves the original audio track in the final output file.

## Features

- **GPU Acceleration**: Utilizes ModernGL (an OpenGL wrapper) for fast rendering.
- **High Performance**: Optimized data preparation on the CPU using OpenCV and NumPy.
- **Full Customization**: Allows adjusting resolution, detail level (grid width), font, and file paths through command-line options.
- **Audio Preservation**: Automatically extracts the audio track from the source video and merges it into the final output using FFmpeg.
- **User-Friendly CLI**: A clean and intuitive command-line interface built with the Click library.
- **Batch Rendering**: Optional batch processing for better GPU utilization.
- **Background Modes**: Choose from `none`, `solid`, `blur`, or `adaptive` backgrounds.
- **Temporary File Option**: Optionally retain the intermediate silent video file.

## How It Works: The GPU Rendering Pipeline

The key to this project's speed is that it avoids slow, pixel-by-pixel processing on the CPU. Instead, it uses a modern graphics pipeline to perform thousands of operations in parallel on the GPU.

Here's a breakdown of the process:

1. **Initialization (Once per run)**

   - A **Character Atlas** is created. This is a single, wide texture (an image) containing all possible ASCII characters (`" .:-=+*#%@"` etc.) laid out side-by-side. This atlas is uploaded to the GPU's memory one time.

2. **Per-Frame Processing (The Loop)**

   - **CPU's Minimal Job**: For each video frame, the CPU performs only very fast operations:
     - It downscales the frame to the target grid size (e.g., 300x168).
     - It determines the color and brightness for each cell in this small grid.
   - **Data Transfer**: The CPU sends a compact package of instructions to the GPU for *every character at once*. This package contains the position, color, and which character to use (as an index) for each cell.
   - **GPU Execution (The Magic)**: The GPU runs two small, highly-optimized programs called **shaders**:
     - **Vertex Shader**: This program runs for each character. Its job is to create a placeholder square (a "quad") and position it correctly on the final output screen.
     - **Fragment Shader**: This program runs for *every single pixel* inside those placeholder squares. It intelligently calculates the final color of the pixel by combining the color data sent from the CPU with the character's shape, which it looks up from the **Character Atlas** texture.
   - **Result**: The entire ASCII frame is rendered instantly into an off-screen buffer (a Framebuffer Object) directly on the GPU.

3. **Final Assembly**

   - The completed frame is read back from the GPU to the CPU.
   - This image is passed to OpenCV, which writes it into a temporary video file.
   - Finally, FFmpeg combines this new silent video with the audio from the original input file to produce the final result.

This pipeline ensures that the most intensive work—drawing tens of thousands of characters per frame—is handled by the hardware best suited for it: the GPU.

## System Requirements

- Python 3.10.4 recommended
- **FFmpeg**: This is a mandatory requirement. FFmpeg must be installed on your system and accessible via the system's `PATH` variable.
  - **Windows**: Download from the [official website](https://ffmpeg.org/download.html) and add the path to the `bin` folder to your `PATH`.
  - **macOS**: `brew install ffmpeg`
  - **Linux**: `sudo apt update && sudo apt install ffmpeg`

## Installation

1. **Clone or download the repository:**

   ```bash
   git clone https://github.com/emeraldsingers/video-to-ascii/
   cd video-to-ascii
   ```

2. **Create and activate a virtual environment (recommended):**

   ```bash
   # Create the environment
   python -m venv venv

   # Activate it
   # Windows
   .\venv\Scripts\activate
   # macOS / Linux
   source venv/bin/activate
   ```

3. **Install the dependencies from **``**:**

   ```bash
   pip install -r requirements.txt
   ```

## Usage

The tool is operated via the command line.

**Basic Command:**

```bash
python ascii_video.py "path/to/your/video.mp4"
```

This command will create a file named `output.mp4` in the current directory with a resolution of 1920x1080.

**Command with all options:**

```bash
python ascii_video.py "C:\videos\my_clip.webm" \
  -o "C:\ascii\final.mp4" \
  -W 1280 -H 720 \
  -g 250 \
  --font "C:\Windows\Fonts\consola.ttf" \
  -bg adaptive \
  -b --batch-size 8 \
  --save-temp
```

### Command Help

To see all available options, run:

```bash
python ascii_video.py --help
```

The output will be similar to this:

```
Usage: ascii_video.py [OPTIONS] INPUT_PATH

  Converts a video file to an ASCII art representation using GPU
  acceleration. The audio from the original file is copied to the final
  output.

Options:
  -o, --output PATH               Output file path.
  -W, --width INTEGER             Width of the output video in pixels.
  -H, --height INTEGER            Height of the output video in pixels.
  -g, --grid-width INTEGER        Number of ASCII characters for the width
                                  (detail level).
  --font FILE                     Path to a MONOSPACED font file (.ttf, .otf).
  --save-temp                     Save the temporary silent video file.
  -b, --use-batch                 Use batch processing for better GPU
                                  utilization.
  --batch-size INTEGER            Number of frames to process in each batch
                                  (only with --use-batch).
  -bg, --background [none|solid|blur|adaptive]
                                  Background mode: none (black), solid (gray),
                                  blur (blurred original), adaptive
                                  (brightness-based)
  -h, --help                      Show this message and exit.
```

## License

This project is licensed under the MIT License.

