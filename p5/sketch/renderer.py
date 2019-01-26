#
# Part of p5: A Python package based on Processing
# Copyright (C) 2017-2018 Abhik Pal
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#
"""The OpenGL renderer for p5."""

import builtins
from contextlib import contextmanager
import math

import numpy as np

import glfw
import OpenGL
from OpenGL.GL import *
from vispy import gloo

import pynanovg as nvg

from ..pmath import matrix
from ..core.primitives import Arc, PShape, Ellipse, Bezier, Rect

##
## Renderer globals.
##
## TODO (2017-08-01 abhikpal):
##
## - Higher level objects *SHOULD NOT* have direct access to internal
##   state variables.
##

## Renderer Globals: USEFUL CONSTANTS
COLOR_WHITE = (1, 1, 1, 1)
COLOR_BLACK = (0, 0, 0, 1)
COLOR_DEFAULT_BG = (0.8, 0.8, 0.8, 1.0)

## Renderer Globals: STYLE/MATERIAL PROPERTIES
##
background_color = COLOR_DEFAULT_BG

fill_color = COLOR_WHITE
fill_enabled = True

stroke_color = COLOR_BLACK
stroke_enabled = True

stroke_weight = 1
stroke_cap = "BUTT"
stroke_join = "MITER"

smooth = True

tint_color = COLOR_BLACK
tint_enabled = False

viewport = None
texture_viewport = None
transform_matrix = np.identity(4)
modelview_matrix = np.identity(4)
projection_matrix = np.identity(4)

## RENDERER SETUP FUNCTIONS.
##
## These don't handle shape rendering directly and are used for setup
## tasks like initialization, cleanup before exiting, resetting views,
## clearing the screen, etc.
##

def initialize_renderer():
    """Initialize the OpenGL renderer.

    For an OpenGL based renderer this sets up the viewport and creates
    the shader programs.

    """
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glEnable(GL_CULL_FACE)
    glDisable(GL_DEPTH_TEST)
    
    global vg
    vg = nvg.Context()

    reset_view()

def clear(color=True, depth=True):
    """Clear the renderer background."""
    gloo.set_state(clear_color=background_color)
    gloo.clear(color=color, depth=depth, stencil=True)

def reset_view():
    """Reset the view of the renderer."""
    global viewport
    global texture_viewport

    global transform_matrix
    global modelview_matrix
    global projection_matrix

    viewport = (
        0,
        0,
        int(builtins.width * builtins.pixel_x_density),
        int(builtins.height * builtins.pixel_y_density),
    )
    texture_viewport = (
        0,
        0,
        builtins.width,
        builtins.height,
    )
    gloo.set_viewport(*viewport)


def cleanup():
    """Run the clean-up routine for the renderer.

    This method is called when all drawing has been completed and the
    program is about to exit.

    """
    pass

## RENDERING FUNTIONS + HELPERS
##
## These are responsible for actually rendring things to the screen.
## For some draw call the methods should be called as follows:
##
##    with draw_loop():
##        # multiple calls to render()
##

def create_texture(data):
    global vg
    pixels = np.array(data).flatten().tobytes()
    return vg.createImageRGBA(data.shape[0], data.shape[1], pixels)

def delete_texture(texture):
    pass

def render_image(image, location, size):
    """Render the image.

    :param image: image to be rendered
    :type image: p5.Image

    :param location: top-left corner of the image
    :type location: tuple | list | p5.Vector

    :param size: target size of the image to draw.
    :type size: tuple | list | p5.Vector
    """
    global vg
    image_pattern = vg.imagePattern(0, 0, size[0], size[1], 0, image._texture, 1)

    apply_transformation()

    vg.beginPath()
    vg.rect(location[0], location[1], size[0], size[1])
    vg.fillPaint(image_pattern)
    vg.fill()

def apply_transformation():
    global vg
    vg.transform(transform_matrix[0][0], 
        transform_matrix[1][0], 
        transform_matrix[0][1], 
        transform_matrix[1][1], 
        transform_matrix[0][3], 
        transform_matrix[1][3])

def read_pixels():
    return gloo.read_pixels(alpha=False)

def flush():
    gloo.flush()

def start_render():
    global vg
    global transform_matrix
    transform_matrix = np.identity(4)
    # clear()
    vg.beginFrame(builtins.width, builtins.height, 1)
    apply_transformation()


def end_render():
    global vg
    vg.endFrame()
    flush()

def draw_shape(shape):
    global vg

    vg.shapeAntiAlias(1 if smooth else 0)
    vg.beginPath()
    vg.save()
    vg.transform(shape._matrix[0][0], 
            shape._matrix[1][0], 
            shape._matrix[0][1], 
            shape._matrix[1][1], 
            shape._matrix[0][3], 
            shape._matrix[1][3])
    if type(shape) is PShape:
        if "point" in shape.attribs:
            x, y = shape.vertices[0]
            vg.shapeAntiAlias(0)
            vg.rect(x, y, 1, 1)
        else:
            i = 0
            for x, y in shape.vertices:
                if i == 0:
                    vg.moveTo(x, y)
                else: 
                    vg.lineTo(x, y)
                i += 1
        
    elif type(shape) is Arc:
        cx, cy, _ = shape._center
        rx, ry, _ = shape._radii
        vg.arc(cx, cy, rx, shape._start_angle, shape._stop_angle, 2)
        if "pie" in shape.attribs:
            vg.lineTo(cx, cy)
    elif type(shape) is Ellipse:
        cx, cy = shape._center
        vg.ellipse(cx, cy, shape._width/2, shape._height/2)
    elif type(shape) is Rect:
        cx, cy = shape._center
        vg.rect(cx, cy, shape._width, shape._height)
    elif type(shape) is Bezier:
        sx, sy = shape._start
        cp1x, cp1y = shape._control_point_1
        cp2x, cp2y = shape._control_point_2
        ex, ey = shape._stop
        vg.moveTo(sx, sy)
        vg.bezierTo(cp1x, cp1y, cp2x, cp2y, ex, ey)

    if "closed" in shape.attribs:
        vg.closePath()
    vg.restore()

    cap_type = 1
    if shape.stroke_cap == "BUTT":
        cap_type = 1
    elif shape.stroke_cap == "ROUND":
        cap_type = 2
    elif shape.stroke_cap == "SQUARE":
        cap_type = 3
    vg.lineCap(cap_type)

    join_type = 1
    if shape.stroke_join == "MITER":
        join_type = 1
    elif shape.stroke_join == "ROUND":
        join_type = 2
    elif shape.stroke_join == "SQUARE":
        join_type = 3
    vg.lineJoin(join_type)
    vg.strokeWidth(shape.stroke_weight)

    if shape.kind == "poly" and shape.fill is not None:
        r, g, b, a = shape.fill.normalized
        vg.fillColor(r, b, g, a)
        vg.fill()
    if shape.stroke is not None:
        r, g, b, a = shape.stroke.normalized
        vg.strokeColor(r, b, g, a)
        vg.stroke()

