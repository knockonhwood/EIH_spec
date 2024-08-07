from kivymd.app import MDApp
from kivymd.uix.filemanager import MDFileManager
from kivy.lang import Builder
from kivy.uix.image import Image
from kivy.uix.camera import Camera
from kivy.uix.label import Label
from kivy.uix.button import Button
import numpy as np
from PIL import Image as pil
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

kv='''
GridLayout:
    rows: 2
    spacing: "48dp"
    padding: "100dp"
    adaptive_height: True
    pos_hint: {"top": 1}

    Button:
        id: file_manager_button
        size: .5,.5
        font_size: '30dp'
        text: 'Select picture file'
        on_release: app.open_file_manager()
    Button:
        id: take_picture_button
        size: .5,.5
        font_size: '30dp'
        text: 'Take picture'
        on_release: app.take_picture()
'''

class FileOpeningApp(MDApp):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.file_manager_obj = MDFileManager(
            select_path=self.select_path,
            exit_manager=self.exit_manager
        )
        self.image = Image()
        self.c_image=Image()
        self.camera = None
        self.img_label = Label(text='No Image')
        self.capture_picture = Button(text='Save Picture', on_release=self.save_pixels)
        self.capture_picture_added = False  # Track if capture button is added
        self.save_transfrm_img_button= Button(text='Save Picture', on_release=self.save_transformed)

    def select_path(self, path):
        print(path)
        self.exit_manager()
        self.file_picture_2_pixels(path)
        self.pixels_2_green_nomalizer()
        self.display_transformed_picture()
        # self.display_image(path)

    def open_file_manager(self):
        self.file_manager_obj.show('/')

    def exit_manager(self):
        self.file_manager_obj.close()

    def display_image(self, path): # button and display manager
        self.image.source = path
        self.root.add_widget(self.image)
        self.root.remove_widget(self.root.ids.file_manager_button)
        self.root.remove_widget(self.root.ids.take_picture_button)

    def take_picture(self): # for camera capture
        self.toggle_camera()
        if self.camera is not None and not self.capture_picture_added:
            self.root.add_widget(self.capture_picture)
            self.capture_picture_added = True  
            # filename = "captured_image.png"  
            # self.camera.export_to_png(filename)
            # self.display_image(filename)
            self.root.remove_widget(self.root.ids.file_manager_button)
            self.root.remove_widget(self.root.ids.take_picture_button)
            self.root.remove_widget(self.image)

    def file_picture_2_pixels(self, path):
         with pil.open(path) as img:
            img = img.convert("RGBA") 
            self.pixels = np.array(img)
            self.size = img.size
            self.root.remove_widget(self.root.ids.file_manager_button)
            self.root.remove_widget(self.root.ids.take_picture_button)

    def build(self):
        return Builder.load_string(kv)
    
    # Functions below control the camera
    
    def toggle_camera(self):
        if self.camera is None:
            print('toggle camera activated')
            self.camera = Camera(resolution=(640, 480), play=True)  
            self.camera.play = True
            self.root.add_widget(self.camera)
        else:
            self.root.remove_widget(self.camera)
            self.camera = None

    def camera_to_pixels(self):
        texture = self.camera.texture
        self.size = texture.size
        self.pixels = np.frombuffer(texture.pixels, np.uint8)
        self.pixels = self.pixels.reshape(self.size[1], self.size[0], 4)  # Reshape to the image size
        self.pixels_2_green_nomalizer()
        self.display_transformed_picture()

    def display_transformed_picture(self):
        temp_image_path = self.pixels_to_image()
        self.c_image.source = temp_image_path
        self.root.add_widget(self.c_image)
        self.root.remove_widget(self.image)
        # self.root.add_widget(self.save_transfrm_img_button)

    def save_pixels(self, *args):
        self.camera_to_pixels()
        if self.capture_picture_added:
            self.capture_picture_added = False
            self.root.remove_widget(self.camera)
            self.root.remove_widget(self.capture_picture)

    # pixel manipulation part

    def pixels_to_image(self): # converts pixels back into image
        pil_img = pil.fromarray(self.transformed_img)
        temp_filename = "temp_image.png"
        pil_img.save(temp_filename)
        return temp_filename
    
    def pixels_2_green_nomalizer(self):
        red, green=self.extract_red_green_intensities(self.pixels)
        norm_data=self.normalizer_funct(red,green)
        self.transformed_img=self.create_color_map_img(norm_data)
        self.save_transformed()
        print('transform complete')

    def extract_red_green_intensities(self, pixels):
        if not (isinstance(pixels, np.ndarray) and pixels.dtype == np.uint8 and pixels.shape[2] == 4):
            raise ValueError("Input must be a numpy array of shape (height, width, 4) with dtype uint8")
        red_intensities = pixels[:, :, 0]
        green_intensities = pixels[:, :, 1]
        return red_intensities, green_intensities
    
    def create_color_map_img(self, norm_data):
        norm = Normalize(vmin=norm_data.min(), vmax=norm_data.max())
        colormap = plt.get_cmap('rainbow')
        scalar_mappable = ScalarMappable(norm=norm, cmap=colormap)
        rgba_image = scalar_mappable.to_rgba(norm_data)
        rgba_image = (rgba_image * 255).astype(np.uint8)
        return rgba_image

    def normalizer_funct(self,red,green):
        with np.errstate(divide='ignore'):
            denominator=np.mean(red)+np.mean(green)
            numerator=red-green
            norm_data = numerator / denominator
            # norm_data[red == 0] = 0
        return norm_data

    def save_transformed(self, *args):  # Modified to accept additional arguments
        pil_img = pil.fromarray(self.transformed_img)
        temp_filename = "transformed_image.png"
        pil_img.save(temp_filename)
        print('saved image')
    
if __name__ == '__main__':
    FileOpeningApp().run()