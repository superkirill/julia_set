import pycuda.driver as drv
import pycuda.tools
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
from pycuda.elementwise import ElementwiseKernel
import tkinter as tk
import random
import numpy as np
from PIL import Image, ImageTk
import cmath


class JuliaFractal(object):
    """Julia Fractal"""
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Julia Fractal")
        self.width = 1000
        self.height = 1000
        self.image = Image.new("RGB", (self.width, self.height))
        self.tk_image = ImageTk.PhotoImage(self.image)
        self.pixel_map = self.image.load()
        self.canvas = tk.Canvas(self.window, width=self.width, height=self.height)
        self.image_on_canvas=self.canvas.create_image(self.width//2, self.height//2, image=self.tk_image)
        self.start_button = tk.Button(self.window, text="Start", command=self.animation_trigger)
        self.auto_button = tk.Button(self.window, text="Auto", command=self.automate)
        self.edit_real = tk.Entry(self.window)
        self.edit_real.insert(0, "0.36")
        self.edit_imaginary = tk.Entry(self.window)
        self.edit_imaginary.insert(0, "0.36")
        self.canvas.grid(row=1, column=0, columnspan=20)
        self.start_button.grid(row=0, column=1)
        tk.Label(self.window, text="c = ").grid(row=0, column=5)
        self.edit_real.grid(row=0, column=6)
        tk.Label(self.window, text=" + ").grid(row=0, column=7)
        self.edit_imaginary.grid(row=0, column=8)
        tk.Label(self.window, text="i").grid(row=0, column=9)
        self.auto_button.grid(row=0, column=17)
        self.is_animated = False
        self.is_automated = False
        # c constant
        self.real = 0.36
        self.imaginary = 0.36
        # Center of a frame
        self.x_center = 0
        self.y_center = 0
        # Field's size
        self.frame_width = 5
        self.frame_height = 5
        # Vector of movement
        self.movement = [0, 0]
        # Maximum number of iterations in a main loop
        self.maxIt = 10000
        self.iterations = 0
        self.colours = np.empty((self.width*self.height,3))
        # Frames borderrs
        self.xa = self.x_center - self.frame_width/2
        self.xb = self.x_center + self.frame_width/2
        self.ya = self.y_center + self.frame_height/2
        self.yb = self.y_center - self.frame_height/2
        # Colours
        self.red = np.empty(self.width * self.height)
        self.green = np.empty(self.width * self.height)
        self.blue = np.empty(self.width * self.height)
        # Julia fractal
        self.complex_gpu = ElementwiseKernel(
            """pycuda::complex<float> *q, int *red, int *green, int *blue,
                int maxiter, float c_real, float c_imaginary""",
            """
            {
                float nreal = 0;
                float real = q[i].real();
                float imag = q[i].imag();
                int required_iterations = 0;
                red[i] = 0;
                green[i] = 0;
                blue[i] = 0;
                for(int curiter = 0; curiter < maxiter; curiter++) {
                    float real2 = real*real;
                    float imag2 = imag*imag;
                    nreal = real2 - imag2 + c_real;
                    imag = 2* real*imag + c_imaginary;
                    real = nreal;
                    required_iterations++;
                    if (real2 + imag2 > 2.0f){
                        int value = curiter % 256;
                        red[i] = abs(80-2* value);
                        green[i] = abs(100-4* value) + curiter % 16;
                        blue[i] = curiter % 32 * 7;
                        break;
                        };
                };

            }
            """,
            "complex5",
            preamble="#include <pycuda-complex.hpp>",)
        self.canvas.bind("<Button-1>", self.update_pos)
        self.canvas.bind("<Button-3>", self.stop_movement)
        self.canvas.bind_all("<MouseWheel>", self.update_size)
        tk.mainloop()

    def automate(self):
        """Start auto correction"""
        self.is_automated = not self.is_automated

    def stop_movement(self, event):
        """Stop all movements"""
        self.movement = [0, 0]

    def update_size(self, event):
        """Resize frame"""
        self.frame_width += event.delta*(self.frame_width)/1000
        self.frame_height += event.delta*(self.frame_height)/1000
        self.xa = self.x_center - self.frame_width/2
        self.xb = self.x_center + self.frame_width/2
        self.ya = self.y_center + self.frame_height/2
        self.yb = self.y_center - self.frame_height/2

    def update_pos(self, event):
        """Update camera's position"""
        x = event.x - self.width//2
        y = event.y - self.height//2
        self.x_center += x*self.frame_width/10000
        self.y_center -= y*self.frame_width/10000
        self.movement[0] += x*self.frame_height/1000000
        self.movement[1] += y*self.frame_height/100000

    def animation_trigger(self):
        """Start or stop animation"""
        self.is_animated = not self.is_animated
        self.start_animation()

    def start_animation(self):
        """Start animation"""
        if self.is_animated:
            if self.is_automated:
                self.imaginary -= 0.0001
                self.edit_real.delete(0,100)
                self.edit_imaginary.delete(0,100)
                self.edit_real.insert(0, str(self.real))
                self.edit_imaginary.insert(0, str(self.imaginary))
            self.draw_fractal()
            self.window.after(1, self.start_animation)

    def gpu_compute_julia_set(self, c):
        """Initialize GPU computations"""
        c_gpu = gpuarray.to_gpu(c.astype(np.complex64))
        iterations_gpu = gpuarray.to_gpu(np.empty(c.shape, dtype=np.int))
        red_gpu = gpuarray.to_gpu(np.empty(c.shape, dtype=np.int))
        green_gpu = gpuarray.to_gpu(np.empty(c.shape, dtype=np.int))
        blue_gpu = gpuarray.to_gpu(np.empty(c.shape, dtype=np.int))
        self.complex_gpu(c_gpu, red_gpu, green_gpu, blue_gpu,
        self.maxIt, np.float32(self.real), np.float32(self.imaginary))
        self.red = red_gpu.get()
        self.green = green_gpu.get()
        self.blue = blue_gpu.get()
        rgbArray = np.zeros((self.width,self.height,3), 'uint8')
        rgbArray[..., 0] = self.red.reshape(self.width, self.height)
        rgbArray[..., 1] = self.green.reshape(self.width, self.height)
        rgbArray[..., 2] = self.blue.reshape(self.width, self.height)
        return rgbArray

    def is_number(self,s):
        """Return True if s is a float number"""
        try:
            float(s)
            return True
        except ValueError:
            return False
    
    def draw_fractal(self):
        """Draw one frame"""
        self.real = float(self.edit_real.get()) if self.is_number(self.edit_real.get()) else 0
        self.imaginary = float(self.edit_imaginary.get()) if self.is_number(self.edit_imaginary.get()) else 0
        self.x_center += self.movement[0]
        self.y_center += self.movement[1]
        self.xa += self.movement[0]
        self.xb += self.movement[0]
        self.ya += self.movement[1]
        self.yb += self.movement[1]
        xx = np.arange(self.xa, self.xb, (self.xb-self.xa)/(self.width))
        if xx.size > self.width:
            x = xx[:-1]
        else:
            x =xx
        yy = np.arange(self.yb, self.ya, (self.ya-self.yb)/(self.height)) * 1j
        if yy.size > self.height:
            y = yy[:-1]
        else:
            y = yy
        y = y.astype(np.complex64)
        c = np.ravel(x+y[:, np.newaxis]).astype(np.complex64)
        colours = self.gpu_compute_julia_set(c)
        self.image = Image.fromarray(colours, mode="RGB")
        self.tk_image = ImageTk.PhotoImage(self.image)
        self.canvas.itemconfig(self.image_on_canvas, image=self.tk_image) 

def main():
    fractal = JuliaFractal()

if __name__ == '__main__':
    main()