"""
2D rendering framework
"""

# pylint: skip-file

import math
import os
import sys

import numpy as np
from gym import error


if 'Apple' in sys.version and 'DYLD_FALLBACK_LIBRARY_PATH' in os.environ:
    os.environ['DYLD_FALLBACK_LIBRARY_PATH'] += ':/usr/lib'

try:
    import pyglet
except ImportError as e:
    raise ImportError(
        """\
    Cannot import pyglet.
    HINT: you can install pyglet directly via 'pip install pyglet'.
    But if you really just want to install all Gym dependencies and not have to think about it,
    'pip install -e .[all]' or 'pip install gym[all]' will do it.
    """
    ) from e

try:
    from pyglet.gl import *
except ImportError as e:
    raise ImportError(
        """\
    Error occurred while running `from pyglet.gl import *`
    HINT: make sure you have OpenGL installed. On Ubuntu, you can run 'apt-get install python-opengl' or 'conda install libglu'.
    If you're running on a server, you may need a virtual frame buffer; something like this should work:
    'xvfb-run -s "-screen 0 1400x900x24" python <your_script.py>'
    """
    ) from e

import pygame

RAD2DEG = 57.29577951308232


def get_display(spec):
    """Convert a display specification (such as :0) into an actual Display
    object.

    Pyglet only supports multiple Displays on Linux.
    """
    if spec is None:
        return pyglet.canvas.get_display()
        # returns already available pyglet_display,
        # if there is no pyglet display available then it creates one
    if isinstance(spec, str):
        return pyglet.canvas.Display(spec)

    raise error.Error(f'Invalid display specification: {spec}. (Must be a string like :0 or None.)')


def get_window(width, height, display, **kwargs):
    """
    Will create a pyglet window from the display specification provided.
    """
    screen = display.get_screens()  # available screens
    config = screen[0].get_best_config()  # selecting the first screen
    context = config.create_context(None)  # create GL context

    return pyglet.window.Window(
        width=width,
        height=height,
        display=display,
        config=config,
        context=context,
        **kwargs,
    )


class Viewer:
    def __init__(self, width, height, display=None):
        pygame.init()
        self.isopen = True

        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        self.window = pygame.Surface((width, height), pygame.SRCALPHA)

        self.geoms: list[Geom] = []
        self.onetime_geoms: list[Geom] = []
        self.transform = Transform()

    def close(self):
        if self.isopen and sys.meta_path:
            # ^^^ check sys.meta_path to avoid 'ImportError: sys.meta_path is None, Python is likely shutting down'
            # self.window.close()
            self.isopen = False
            pygame.quit()

    def window_closed_by_user(self):
        self.isopen = False

    def set_bounds(self, left, right, bottom, top):
        assert right > left and top > bottom
        scalex = self.width / (right - left)
        scaley = self.height / (top - bottom)
        self.transform = Transform(
            translation=(-left * scalex, -bottom * scaley), scale=(scalex, scaley)
        )

        print(self.transform.translation)
        print(self.transform.scale)

    def add_geom(self, geom):
        self.geoms.append(geom)

    def add_onetime(self, geom):
        self.onetime_geoms.append(geom)

    def render(self, return_rgb_array=False) -> tuple[np.ndarray, bool]:
        self.screen.fill((255, 255, 255, 255))

        for geom in self.geoms:
            geom.add_transform(self.transform)
            geom.render(self.screen)
            geom.sub_transform(self.transform)
            pass

        for geom in self.onetime_geoms:
            geom.add_transform(self.transform)
            geom.render(self.screen)
            geom.sub_transform(self.transform)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.isopen = False
        
        # self.screen.blit(self.window, (0, 0))
        
        pygame.display.flip()

        return self.isopen

        # self.transform.disable()
        # arr = None
        # if return_rgb_array:
        #     buffer = pyglet.image.get_buffer_manager().get_color_buffer()
        #     image_data = buffer.get_image_data()
        #     arr = np.asarray(image_data.get_data(), dtype=np.uint8)
        #     # In https://github.com/openai/gym-http-api/issues/2, we
        #     # discovered that someone using Xmonad on Arch was having
        #     # a window of size 598 x 398, though a 600 x 400 window
        #     # was requested. (Guess Xmonad was preserving a pixel for
        #     # the boundary.) So we use the buffer height/width rather
        #     # than the requested one.
        #     arr = arr.reshape(buffer.height, buffer.width, 4)
        #     arr = arr[::-1, :, 0:3]
        # self.window.flip()
        # self.onetime_geoms = []

        # try:
            # return arr, self.isopen
        # except:
            # return arr, False


    def __del__(self):
        self.close()


def _add_attrs(geom, attrs):
    if 'color' in attrs:
        geom.set_color(*attrs['color'])
    if 'linewidth' in attrs:
        geom.set_linewidth(attrs['linewidth'])


class Geom:
    def __init__(self):
        self.color = pygame.Color(0, 0, 0, 255)
        self.transform = Transform()
        self.attrs: dict = {}

        self.v = []
        self.V = []
        self.width = 0


    def render(self, surf: pygame.Surface):
        max_x = max(self.v, key= lambda x: x[0])[0]
        max_y = max(self.v, key= lambda x: x[1])[1]
        min_x = min(self.v, key= lambda x: x[0])[0]
        min_y = min(self.v, key= lambda x: x[1])[1]


        s = pygame.Surface((max_x - min_x+1 + self.width, max_y - min_y+1 + self.width), pygame.SRCALPHA)
        self.V = [[v[0]-min_x, v[1]-min_y] for v in self.v]

        self.render1(s)

        # pygame.draw.polygon(s, self.color, V, self.linewidth, **self.attrs)
        
        if self.transform.scale[0] != 1 or self.transform.scale[1] != 1:
            s = pygame.transform.scale_by(s, self.transform.scale)
        
        if self.transform.rotation != 0:
            s = pygame.transform.rotate(s, self.transform.rotation)

        x = self.transform.translation[0] + min_x*self.transform.scale[0]
        y = self.transform.translation[1] + min_y*self.transform.scale[1]

        # print(x, y)

        # x = 1
        # y = 1


        surf.blit(s, (x, y))

    def render1(self, surf: pygame.Surface):
        raise NotImplementedError

    def add_attr(self, key, value):
        self.attrs[key] = value
        # self.attrs.append(attr)

    def set_color(self, r, g, b, a=1):
        r = self.toRGB(r)
        g = self.toRGB(g)
        b = self.toRGB(b)
        a = self.toRGB(a)
        self.color.update(r, g, b, a)
    
    def set_alpha(self, a=1):
        a = self.toRGB(a)
        self.color.a = a

    
    @staticmethod
    def toRGB(value):
        if value <= 1 and type(value) == float:
            return int(value*255)
        return value
    
    def add_transform(self, transform):
        self.transform.add(transform)
    
    def sub_transform(self, transform):
        self.transform.sub(transform)


class Attr:
    def enable(self):
        raise NotImplementedError

    def disable(self):
        pass


class Transform(Attr):
    def __init__(self, translation=(0.0, 0.0), rotation=0.0, scale=(1, 1)):
        self.set_translation(*translation)
        self.set_rotation(rotation)
        self.set_scale(*scale)

    def enable(self):
        pass

    def disable(self):
        pass

    def set_translation(self, newx, newy):
        self.translation = (float(newx), float(newy))

    def set_rotation(self, new):
        self.rotation = float(new)

    def set_scale(self, newx, newy):
        self.scale = (float(newx), float(newy))

    
    def add(self, transform):
        self.set_translation(self.translation[0] + transform.translation[0], self.translation[1] + transform.translation[1])
        self.set_rotation(self.rotation + transform.rotation)
        self.set_scale(self.scale[0] * transform.scale[0], self.scale[1] * transform.scale[1])
    
    def sub(self, transform):
        self.set_translation(self.translation[0] - transform.translation[0], self.translation[1] - transform.translation[1])
        self.set_rotation(self.rotation - transform.rotation)
        self.set_scale(self.scale[0] / transform.scale[0], self.scale[1] / transform.scale[1])
    

class Point(Geom):
    def __init__(self):
        Geom.__init__(self)

    def render(self, surf: pygame.Surface):
        pass
        # glBegin(GL_POINTS)  # draw point
        # glVertex3f(0.0, 0.0, 0.0)
        # glEnd()


class FilledPolygon(Geom):
    def __init__(self, v):
        Geom.__init__(self)
        self.v = v

    def render1(self, surf: pygame.Surface):
        pygame.draw.polygon(surf, self.color, self.V, **self.attrs)

class PolyLine(Geom):
    def __init__(self, v, close):
        Geom.__init__(self)
        self.v = v
        self.close = close
        self.width = 1

    def render1(self, surf: pygame.Surface):
        pygame.draw.polygon(surf, self.color, self.V, self.width, **self.attrs)


    def set_linewidth(self, x):
        self.width = x

class Compound(Geom):
    def __init__(self, gs: list[Geom]):
        Geom.__init__(self)
        self.gs = gs

    def render(self, surf: pygame.Surface):
        for g in self.gs:
            g.render(surf)




class Image(Geom):
    def __init__(self, fname, width, height):
        Geom.__init__(self)
        self.set_color(1.0, 1.0, 1.0)
        self.width = width
        self.height = height
        # img = pyglet.image.load(fname)
        img = pygame.image.load(fname)
        self.img = pygame.transform.scale(img, (width, height))
        self.flip = False
        self.v = [[-width//2, -height//2], [width//2, -height//2], [width//2, height//2], [-width//2, -height//2]]
        self.V = []
    

    def render1(self, surf: pygame.Surface):
        surf.blit(self.img, (self.V[0], self.V[1]))
        # self.img.blit(-self.width / 2, -self.height / 2, width=self.width, height=self.height)

def make_circle(radius=10, res=30, filled=True):
    points = []
    for i in range(res):
        ang = 2 * math.pi * i / res
        points.append((math.cos(ang) * radius, math.sin(ang) * radius))
    if filled:
        return FilledPolygon(points)
    return PolyLine(points, True)


def make_polygon(v, filled=True):
    if filled:
        return FilledPolygon(v)
    return PolyLine(v, True)


def make_polyline(v):
    return PolyLine(v, False)


def make_capsule(length, width):
    l, r, t, b = 0, length, width / 2, -width / 2
    box = make_polygon([(l, b), (l, t), (r, t), (r, b)])
    circ0 = make_circle(width / 2)
    circ1 = make_circle(width / 2)
    circ1.add_transform(Transform(translation=(length, 0)))
    geom = Compound([box, circ0, circ1])
    return geom




def main():
    a = Viewer(800, 800)
    bound = 1.05 * 1000

    a.set_bounds(-bound, bound, -bound, bound)


    margin = make_polygon(
        1000 * np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]]), filled=False
    )

    margin.set_linewidth(3)
    a.add_geom(margin)


    run = True
    while run:
        run = a.render()
        # pygame.draw.polygon(a.screen, [255, 0, 0], [[10, 100], [100, 100], [100, 200]])
    
    pygame.quit()

if __name__ == "__main__":
    main()
