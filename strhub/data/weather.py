"""
Scene text may be captured under different weather conditions:
1) Fog, 
2) Snow, 
3) Frost, 
4) Rain 
and 
5) Shadow.

Fog, Snow, Frost Reference: https://github.com/hendrycks/robustness
Hacked together for STR/Copyright 2021 by Rowel Atienza
"""

import math
from io import BytesIO

import cv2
import numpy as np
from PIL import Image, ImageOps, ImageDraw
from pkg_resources import resource_filename

from .ops import plasma_fractal


class Fog:
    def __init__(self, rng=None, prob=0.5):
        self.rng = np.random.default_rng() if rng is None else rng
        self.prob=prob
        self.mag = np.random.randint(-1, 3)

    def __call__(self, img):
        if self.rng.uniform(0, 1) > self.prob:
            return img

        w, h = img.size
        c = [(1.5, 2), (2., 2), (2.5, 1.7)]
        if self.mag < 0 or self.mag >= len(c):
            index = self.rng.integers(0, len(c))
        else:
            index = self.mag
        c = c[index]

        n_channels = len(img.getbands())
        isgray = n_channels == 1

        img = np.asarray(img) / 255.
        max_val = img.max()
        # Make sure fog image is at least twice the size of the input image
        max_size = 2 ** math.ceil(math.log2(max(w, h)) + 1)
        fog = c[0] * plasma_fractal(mapsize=max_size, wibbledecay=c[1], rng=self.rng)[:h, :w][..., np.newaxis]
        # x += c[0] * plasma_fractal(wibbledecay=c[1])[:224, :224][..., np.newaxis]
        # return np.clip(x * max_val / (max_val + c[0]), 0, 1) * 255
        if isgray:
            fog = np.squeeze(fog)
        else:
            fog = np.repeat(fog, 3, axis=2)

        img += fog
        img = np.clip(img * max_val / (max_val + c[0]), 0, 1) * 255
        return Image.fromarray(img.astype(np.uint8))


class Frost:
    def __init__(self, rng=None, prob=0.5):
        self.rng = np.random.default_rng() if rng is None else rng
        self.prob=prob
        self.mag = np.random.randint(-1, 3)
        
    def __call__(self, img):
        if self.rng.uniform(0, 1) > self.prob:
            return img

        w, h = img.size
        c = [(0.78, 0.22), (0.64, 0.36), (0.5, 0.5)]
        if self.mag < 0 or self.mag >= len(c):
            index = self.rng.integers(0, len(c))
        else:
            index = self.mag
        c = c[index]

        filename = [resource_filename(__name__, 'frost/frost1.png'),
                    resource_filename(__name__, 'frost/frost2.png'),
                    resource_filename(__name__, 'frost/frost3.png'),
                    resource_filename(__name__, 'frost/frost4.jpg'),
                    resource_filename(__name__, 'frost/frost5.jpg'),
                    resource_filename(__name__, 'frost/frost6.jpg')]
        index = self.rng.integers(0, len(filename))
        filename = filename[index]
        # Some images have transparency. Remove alpha channel.
        frost = Image.open(filename).convert('RGB')

        # Resize the frost image to match the input image's dimensions
        f_w, f_h = frost.size
        if w / h > f_w / f_h:
            f_h = round(f_h * w / f_w)
            f_w = w
        else:
            f_w = round(f_w * h / f_h)
            f_h = h
        frost = np.asarray(frost.resize((f_w, f_h)))

        # randomly crop
        y_start, x_start = self.rng.integers(0, f_h - h + 1), self.rng.integers(0, f_w - w + 1)
        frost = frost[y_start:y_start + h, x_start:x_start + w]

        n_channels = len(img.getbands())
        isgray = n_channels == 1

        img = np.asarray(img)

        if isgray:
            img = np.expand_dims(img, axis=2)
            img = np.repeat(img, 3, axis=2)

        img = np.clip(np.round(c[0] * img + c[1] * frost), 0, 255)
        img = Image.fromarray(img.astype(np.uint8))
        if isgray:
            img = ImageOps.grayscale(img)

        return img


class Rain:
    def __init__(self, rng=None):
        self.rng = np.random.default_rng() if rng is None else rng

    def __call__(self, img, mag=-1, prob=1.):
        if self.rng.uniform(0, 1) > prob:
            return img

        img = img.copy()
        w, h = img.size
        n_channels = len(img.getbands())
        isgray = n_channels == 1
        line_width = self.rng.integers(1, 2)

        c = [50, 70, 90]
        if mag < 0 or mag >= len(c):
            index = 0
        else:
            index = mag
        c = c[index]

        n_rains = self.rng.integers(c, c + 20)
        slant = self.rng.integers(-60, 60)
        fillcolor = 200 if isgray else (200, 200, 200)

        draw = ImageDraw.Draw(img)
        max_length = min(w, h, 10)
        for i in range(1, n_rains):
            length = self.rng.integers(5, max_length)
            x1 = self.rng.integers(0, w - length)
            y1 = self.rng.integers(0, h - length)
            x2 = x1 + length * math.sin(slant * math.pi / 180.)
            y2 = y1 + length * math.cos(slant * math.pi / 180.)
            x2 = int(x2)
            y2 = int(y2)
            draw.line([(x1, y1), (x2, y2)], width=line_width, fill=fillcolor)

        return img


class Shadow:
    def __init__(self, rng=None):
        self.rng = np.random.default_rng() if rng is None else rng

    def __call__(self, img, mag=-1, prob=1.):
        if self.rng.uniform(0, 1) > prob:
            return img

        # img = img.copy()
        w, h = img.size
        n_channels = len(img.getbands())
        isgray = n_channels == 1

        c = [64, 96, 128]
        if mag < 0 or mag >= len(c):
            index = 0
        else:
            index = mag
        c = c[index]

        img = img.convert('RGBA')
        overlay = Image.new('RGBA', img.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)
        transparency = self.rng.integers(c, c + 32)
        x1 = self.rng.integers(0, w // 2)
        y1 = 0

        x2 = self.rng.integers(w // 2, w)
        y2 = 0

        x3 = self.rng.integers(w // 2, w)
        y3 = h - 1

        x4 = self.rng.integers(0, w // 2)
        y4 = h - 1

        draw.polygon([(x1, y1), (x2, y2), (x3, y3), (x4, y4)], fill=(0, 0, 0, transparency))

        img = Image.alpha_composite(img, overlay)
        img = img.convert("RGB")
        if isgray:
            img = ImageOps.grayscale(img)

        return img
