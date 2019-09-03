"""
James McDermott james.mcdermott@nuigalway.ie

This module provides a type of interpolating GUI for controlling multiple
float parameters at once. For 10 parameters, we get 5 points arranged in a
2D space. We can grab any of them and move it around in 2D. We can also click
anywhere and the currently active one will move there. There are buttons
to show the current positions as coordinates and to show which is active.

It seems to work ok, but several ugly aspects of the design:

* The Interpolate widget should be a RelativeWidget, and calculate positions
  in widget-relative terms.

* Each of the widgets (ColorButtonGrid, InterpolateWidget) uses 
  self.parent.XXX to "reach inside" the other.

"""



import random
import math
import numpy as np

from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.relativelayout import RelativeLayout
from kivy.uix.checkbox import CheckBox
from kivy.uix.behaviors import ToggleButtonBehavior
from kivy.properties import BooleanProperty
from kivy.core.window import Window
from kivy.graphics import Color, Ellipse, Line, Rectangle
from kivy.graphics.texture import Texture
from kivy.clock import Clock



def HSVtoRGB(c):
    # https://stackoverflow.com/a/40986913/86465
    return tuple(Color(*c, mode='hsv').rgba)

def color(i, n):
    return (i / n, 0.5, 0.8, 1.0)

spacing = 10

class ControlButtonGrid(GridLayout):
    def __init__(self, cols, *args, **kwargs):
        super().__init__(cols=cols, *args, **kwargs)
        n = cols
        save_button = Button(text="Save",
                             #background_normal="",
                             background_down="",
                             color=(0, 0, 0, 1))
        save_button.bind(on_press=self.save)
        rand_button = Button(text="Randomise",
                             #background_normal="",
                             background_down="",
                             color=(0, 0, 0, 1))
        rand_button.bind(on_press=self.randomise)
        self.buttons = [
            save_button, rand_button
        ]
        for b in self.buttons: self.add_widget(b)

    def save(self, instance):
        print(instance, "is pressed with state", instance.state)
        print("But saving is not implemented!")
        # TODO save positions
        # TODO save self.parent.drum_grid.data
    def randomise(self, instance):
        # print(instance, "is pressed with state", instance.state)
        self.parent.interp.randomise_positions()
        self.parent.interp.draw()


class ColorButtonGrid(GridLayout):
    def __init__(self, cols, *args, **kwargs):
        super().__init__(cols=cols, *args, **kwargs)
        n = cols
        self.buttons = []
        for i in range(n):
            b = ToggleButton(text="    ",
                             color=(1, 1, 1, 1),
                             background_normal="",
                             background_down="",
                             background_color=HSVtoRGB(color(i, n)),
                             group="my_toggles",
                             allow_no_selection=False
            )
            b.idx = i
            b.bind(state=self.on_state)
            self.buttons.append(b)
            self.add_widget(b)
        self.buttons[0].state = "down"
        self.which_active = 0

    def on_state(self, instance, value):
        #print(instance.idx, "is pressed with value", value, "and state", instance.state)
        if instance.state == "down":
            self.which_active = instance.idx
            if not instance.text.startswith("** "):
                instance.text = "** " + instance.text
        else:
            #print("removing **", instance.idx)
            while instance.text.startswith("** "):
                instance.text = instance.text[3:]
        # print([b.state for b in self.buttons])
        
class InterpolateContainerWidget(BoxLayout):
    def __init__(self, n, **kwargs):
        super().__init__(orientation="vertical", **kwargs)
        self.n = n
        self.spacing = spacing
        self._keyboard = Window.request_keyboard(
            self._keyboard_closed, self, 'text')
        self._keyboard.bind(on_key_down=self._on_keyboard_down)

        self.interp = InterpolateWidget(n, size_hint=(1, 1 / (1 + (5 * (grid_prop + 3 * spacing / s)))))
        self.grid = ColorButtonGrid(rows=1, cols=n, spacing=spacing, size_hint=(1, grid_prop / (1 + (5 * (grid_prop + 3 * spacing / s)))))
        self.control_grid = ControlButtonGrid(cols=2, spacing=spacing, size_hint=(1, grid_prop / (1 + (5 * (grid_prop + 3 * spacing / s)))))
        self.drum_grid = DrumGridWidget(cols=64, rows=9, size_hint=(1, 3 * grid_prop / (1 + (5 * (grid_prop + 3 * spacing / s)))))
        self.add_widget(self.interp)
        self.add_widget(self.grid)
        self.add_widget(self.control_grid)
        self.add_widget(self.drum_grid)
        self.interp.draw()

    def _keyboard_closed(self):
        # print('My keyboard have been closed!')
        self._keyboard.unbind(on_key_down=self._on_keyboard_down)
        self._keyboard = None

    def _on_keyboard_down(self, keyboard, keycode, text, modifiers):
        # print('The key', keycode, 'have been pressed')
        # print(' - text is %r' % text)
        # print(' - modifiers are %r' % modifiers)

        if 'ctrl' in modifiers and keycode[1] == 'c':
            App.get_running_app().stop()

        idx = -1
        try:
            idx = int(text) - 1 # convert from [1, n] to [0, n-1]
        except:
            pass
        if 0 <= idx < self.n:
            # print("going to set", idx, "down")
            for b in self.grid.buttons:
                if b.idx == idx:
                    b.state = 'down'
                else:
                    b.state = 'normal'
        
        # Return True to accept the key. Otherwise, it will be used by
        # the system.
        return True


def dist(a, b, c, d):
    return math.sqrt((a-c)**2 + (b-d)**2)

# class InterpolateWidget(RelativeLayout):
class InterpolateWidget(Widget):

    def __init__(self, n, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n = n
        self.randomise_positions()
        
    def randomise_positions(self):
        self.outputs = [(random.random(), random.random()) for i in range(self.n)]
        self.positions = [(o[0] * s, o[1] * s + 3 * spacing + (5 * s * grid_prop)) for o in self.outputs]
        assert all(5 * s * grid_prop + 3 * spacing <= pos[1] <= 5 * s * grid_prop + 3 * spacing + s * (1 + grid_prop) for pos in self.positions)
        
    def draw(self):
        for i, output in enumerate(self.outputs):
            txt = "{:.2f}, {:.2f}".format(*output)
            if i == self.parent.grid.which_active:
                txt = "** " + txt
            self.parent.grid.buttons[i].text = txt
        self.canvas.clear()            
        with self.canvas:
            for i, pos in enumerate(self.positions):
                Color(*color(i, len(self.positions)), mode="hsv")
                d = 30.
                pos = (pos[0] - d / 2, pos[1] - d / 2)
                Rectangle(pos=pos, size=(d, d))
        
    def on_touch_down(self, touch):
        sizey = self.size[1]
        if touch.y < sizey * 5 * grid_prop + 3 * spacing:
            return # touch outside the interp widget
        mindist = 20
        idx = -1
        for i, (xi, yi) in enumerate(self.positions):
            d = dist(touch.x, touch.y, xi, yi)
            if d < mindist:
                idx, mindist = i, d
        if idx != -1:
            for i in range(len(self.positions)):
                if i == idx:
                    self.parent.grid.buttons[i].state = "down"
                else:
                    self.parent.grid.buttons[i].state = "normal"
        self.on_touch_move(touch)

    def on_touch_move(self, touch):
        idx = self.parent.grid.which_active
        # print(idx)
        sizex = self.size[0]
        sizey = self.size[1]
        # print("sizex, sizey, touchx, touchy", sizex, sizey, touch.x, touch.y)        
        if touch.y < sizey * 5 * grid_prop + 3 * spacing:
            # print("touch.y was too small", touch.y,  sizey * 5 * grid_prop + 3 * spacing)
            touch.y = sizey * 5 * grid_prop + 3 * spacing
        if touch.y >= sizey * (1 + 5 * grid_prop) + 3 * spacing:
            # print("touch.y was too big", touch.y,  sizey * (1 + 5 * grid_prop) + 3 * spacing)
            touch.y = sizey * (1 + 5 * grid_prop) + 3 * spacing - 1
        if touch.x < 0:
            touch.x = 0
        if touch.x >= sizex:
            touch.x = sizex - 1
        xy = (touch.x, touch.y)
        self.canvas.clear()
        self.positions[idx] = xy
        #print(sizey, s, xy[0], xy[1])
        self.outputs[idx] = xy[0] / (sizex - 1), (xy[1] - 3 * spacing - 5 * grid_prop * sizey) / sizey
        self.draw()

class DrumGridWidget(Widget):
    def __init__(self, rows, cols, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rows = rows 
        self.cols = cols
        self.texture = Texture.create(size=(64, 9))
        self.texture.mag_filter = "nearest"
        self.draw()        
        
    def set_data(self, t, data):
        s0, s1 = self.rows, self.cols
        x = np.repeat(data, 3)
        x = x.reshape((s0, s1, 3))
        x[:, t, 1] += 50 # highlight in green
        x = x.reshape(64 * 9 * 3)
        self.texture.blit_buffer(x, colorfmt='rgb', bufferfmt='ubyte')
        self.canvas.ask_update()
        
    def draw(self):
        with self.canvas:
            Rectangle(texture=self.texture, pos=self.pos, size=(s, s*3*grid_prop))
                

class InterpolateApp(App):
    def __init__(self, n_controllers=3, clock_cb=None, timeout=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        Window.size = (s, int(s*(1 + 5 * grid_prop) + 3 * spacing))
        if clock_cb:
            Clock.schedule_interval(clock_cb, timeout)
        self.n_controllers = n_controllers

    def build(self):
        self.layout = BoxLayout()
        self.interp = InterpolateContainerWidget(self.n_controllers)
        #self.interp2 = InterpolateContainerWidget(self.n_controllers)
        self.layout.add_widget(self.interp)
        #self.layout.add_widget(self.interp2)
        return self.layout


s = 750
grid_prop = 0.05

if __name__ == '__main__':
    app = InterpolateApp()
    app.run()
