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
import builtins

from ..sketch import renderer
from .color import Color
from .color import color_mode
from .image import image
from .image import image_mode
from .image import PImage
from .primitives import rect
from .structure import push_style
from .transforms import push_matrix
from .transforms import reset_transforms

__all__ = [ 'background', 'fill', 'no_fill',
            'stroke', 'stroke_weight', 'stroke_cap', 'stroke_join', 'no_stroke', 'tint', 'no_tint', 'smooth', 'no_smooth' ]

def fill(*fill_args, **fill_kwargs):
    """Set the fill color of the shapes.

    :param fill_args: positional arguments to be parsed as a color.
    :type fill_args: tuple

    :param fill_kwargs: keyword arguments to be parsed as a color.
    :type fill_kwargs: dict

    :returns: The fill color.
    :rtype: Color

    """
    fill_color = Color(*fill_args, **fill_kwargs)
    renderer.fill_enabled = True
    renderer.fill_image_enabled = False
    renderer.fill_color = fill_color.normalized
    return fill_color

def no_fill():
    """Disable filling geometry."""
    renderer.fill_enabled = False

def stroke(*color_args, **color_kwargs):
    """Set the color used to draw lines around shapes

    :param color_args: positional arguments to be parsed as a color.
    :type color_args: tuple

    :param color_kwargs: keyword arguments to be parsed as a color.
    :type color_kwargs: dict

    :note: Both color_args and color_kwargs are directly sent to
        Color.parse_color

    :returns: The stroke color.
    :rtype: Color
    """
    stroke_color = Color(*color_args, **color_kwargs)
    renderer.stroke_enabled = True
    renderer.stroke_color = stroke_color.normalized

def stroke_weight(weight):
    """Set the width of the lines around shapes

    :param weight: width of the stroke.
    :type weight: float
    """
    renderer.stroke_weight = weight

def stroke_cap(cap):
    """Set the end (cap) of the stroke is drawn

    :param cap: type of the cap.
    :type cap: string
    """

    if cap != "BUTT" and cap != "ROUND" and cap != "SQUARE":
        raise ValueError("Cap must be \"BUTT\", \"ROUND\" or \"SQUARE\"")

    renderer.stroke_cap = cap

def stroke_join(join):
    """Set the corners of the stroke are drawn

    :param join: type of join.
    :type weijoinght: string
    """

    if join != "MITER" and join != "ROUND" and join != "SQUARE":
        raise ValueError("Join must be \"MITER\", \"ROUND\" or \"SQUARE\"")

    renderer.stroke_join = join

def no_stroke():
    """Disable drawing the stroke around shapes."""
    renderer.stroke_enabled = False

def tint(*color_args, **color_kwargs):
    """Set the tint color for the sketch.

    :param color_args: positional arguments to be parsed as a color.
    :type color_args: tuple

    :param color_kwargs: keyword arguments to be parsed as a color.
    :type color_kwargs: dict

    :note: Both color_args and color_kwargs are directly sent to
        Color.parse_color

    :returns: The tint color.
    :rtype: Color
    """
    tint_color = Color(*color_args, **color_kwargs)
    renderer.tint_enabled = True
    renderer.tint_color = tint_color.normalized

def no_tint():
    """Disable tinting of images."""
    renderer.tint_enabled = False

def smooth():
    """Enable smoothing."""
    renderer.smooth = True

def no_smooth():
    """Disable smoothing."""
    renderer.smooth = False

def background(*args, **kwargs):
    """Set the background color for the renderer.

    :param args: positional arguments to be parsed as a color.
    :type color_args: tuple

    :param kwargs: keyword arguments to be parsed as a color.
    :type kwargs: dict

    :note: Both args and color_kwargs are directly sent to
        color.parse_color

    :note: When setting an image as the background, the dimensions of
        the image should be the same as that of the sketch window.

    :returns: The background color or image.
    :rtype: p5.Color | p5.PImage

    :raises ValueError: When the dimensions of the image and the
        sketch do not match.

    """
    if len(args) == 1 and isinstance(args[0], PImage):
        background_image = args[0]
        sketch_size = (width, height)

        if sketch_size != background_image.size:
            msg = "Image dimension {} and sketch dimension {} do not match"
            raise ValueError(msg.format(background_image.size, sketch_size))

        with push_style():
            no_tint()
            image_mode('corner')
            with push_matrix():
                image(background_image, (0, 0))

        return background_image

    with push_style():
        background_color = Color(*args, **kwargs)
        fill(background_color)
        no_stroke()

        with push_matrix():
            reset_transforms()
            rect((0, 0), builtins.width, builtins.height, mode='CORNER')
            renderer.background_color = background_color.normalized
