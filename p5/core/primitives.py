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
from collections import namedtuple
import functools
import math
from math import sin
from math import cos
from math import radians

import numpy as np

from .. import sketch

from ..pmath import Point
from ..pmath import curves
from ..pmath import remap
from ..pmath.utils import SINCOS
from ..pmath.utils import SINCOS_PRECISION

from .shape import PShape

__all__ = ['point', 'line', 'arc', 'triangle', 'quad',
           'rect', 'square', 'circle', 'ellipse', 'ellipse_mode',
           'rect_mode', 'bezier', 'curve', 'create_shape', 'draw_shape']

_rect_mode = 'CORNER'
_ellipse_mode = 'CENTER'
_shape_mode = 'CORNER'

# We use these in ellipse tessellation. The algorithm is similar to
# the one used in Processing and the we compute the number of
# subdivisions per ellipse using the following formula:
#
#    min(M, max(N, (2 * pi * size / F)))
#
# Where,
#
# - size :: is the measure of the dimensions of the circle when
#   projected in screen coordiantes.
#
# - F :: sets the minimum number of subdivisions. A smaller `F` would
#   produce more detailed circles (== POINT_ACCURACY_FACTOR)
#
# - N :: Minimum point accuracy (== MIN_POINT_ACCURACY)
#
# - M :: Maximum point accuracy (== MAX_POINT_ACCURACY)
#
MIN_POINT_ACCURACY = 20
MAX_POINT_ACCURACY = 200
POINT_ACCURACY_FACTOR = 10

def _draw_on_return(func):
    """Set shape parameters to default renderer parameters

    """
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        s = func(*args, **kwargs)
        draw_shape(s)
        return s

    return wrapped

class Arc(PShape):
    def __init__(self, center, radii, start_angle, stop_angle,
                 attribs='open pie', **kwargs):
        self._center = center
        self._radii = radii
        self._start_angle = start_angle
        self._stop_angle = stop_angle

        super().__init__(vertices=[], attribs=attribs, **kwargs)

class Rect(PShape):
    def __init__(self, center, width, height, **kwargs):
        self._center = center
        self._width = width
        self._height = height

        super().__init__(vertices=[], **kwargs)

class Ellipse(PShape):
    def __init__(self, center, width, height, **kwargs):
        self._center = center
        self._width = width
        self._height = height

        super().__init__(vertices=[], **kwargs)

class Bezier(PShape):
    def __init__(self, start, control_point_1, control_point_2, stop, **kwargs):
        self._start = start
        self._control_point_1 = control_point_1
        self._control_point_2 = control_point_2
        self._stop = stop

        super().__init__(vertices=[], attribs='path', **kwargs)

@_draw_on_return
def point(x, y, z=0):
    """Returns a point.

    :param x: x-coordinate of the shape.
    :type x: int or float

    :param y: y-coordinate of the shape.
    :type y: int or float

    :param z: z-coordinate of the shape (defaults to 0).
    :type z: int or float

    :returns: A point PShape.
    :rtype: PShape

    """
    return PShape([(x, y)], attribs='point')

@_draw_on_return
def line(p1, p2):
    """Returns a line.

    :param p1: Coordinates of the starting point of the line.
    :type p1: tuple

    :param p2: Coordinates of the end point of the line.
    :type p2: tuple

    :returns: A line PShape.
    :rtype: PShape

    """
    path = [
        Point(*p1),
        Point(*p2)
    ]
    return PShape(path, attribs='path')

@_draw_on_return
def bezier(start, control_point_1, control_point_2, stop):
    """Return a bezier path defined by two control points.

    :param start: The starting point of the bezier curve.
    :type start: tuple.

    :param control_point_1: The first control point of the bezier
        curve.
    :type control_point_1: tuple.

    :param control_point_2: The second control point of the bezier
        curve.
    :type control_point_2: tuple.

    :param stop: The end point of the bezier curve.
    :type stop: tuple.

    :returns: A bezier path.
    :rtype: PShape.

    """
    return Bezier(start, control_point_1, control_point_2, stop)

@_draw_on_return
def curve(point_1, point_2, point_3, point_4):
    """Return a Catmull-Rom curve defined by four points.

    :param point_1: The first point of the curve.
    :type point_1: tuple

    :param point_2: The first point of the curve.
    :type point_2: tuple

    :param point_3: The first point of the curve.
    :type point_3: tuple

    :param point_4: The first point of the curve.
    :type point_4: tuple

    :returns: A curved path.
    :rtype: PShape

    """
    vertices = []
    steps = curves.curve_resolution
    for i in range(steps + 1):
        t = i / steps
        p = curves.curve_point(point_1, point_2, point_3, point_4, t)
        vertices.append(p[:3])

    return PShape(vertices, attribs='path')

@_draw_on_return
def triangle(p1, p2, p3):
    """Return a triangle.

    :param p1: coordinates of the first point of the triangle
    :type p1: tuple | list | p5.Vector

    :param p2: coordinates of the second point of the triangle
    :type p2: tuple | list | p5.Vector

    :param p3: coordinates of the third point of the triangle
    :type p3: tuple | list | p5.Vector

    :returns: A triangle.
    :rtype: p5.PShape
    """
    tr = PShape()
    with tr.edit():
        for pt in [p1, p2, p3]:
            tr.add_vertex(pt)
    return tr

@_draw_on_return
def quad(p1, p2, p3, p4):
    """Return a quad.

    :param p1: coordinates of the first point of the quad
    :type p1: tuple | list | p5.Vector

    :param p2: coordinates of the second point of the quad
    :type p2: tuple | list | p5.Vector

    :param p3: coordinates of the third point of the quad
    :type p3: tuple | list | p5.Vector

    :param p4: coordinates of the fourth point of the quad
    :type p4: tuple | list | p5.Vector

    :returns: A quad.
    :rtype: PShape
    """
    qd = PShape()
    with qd.edit():
        for pt in [p1, p2, p3, p4]:
            qd.add_vertex(pt)
    return qd

@_draw_on_return
def rect(coordinate, *args, mode=None):
    """Return a rectangle.

    :param coordinate: Represents the lower left corner of then
        rectangle when mode is 'CORNER', the center of the rectangle
        when mode is 'CENTER' or 'RADIUS', and an arbitrary corner
        when mode is 'CORNERS'

    :type coordinate: tuple | list | p5.Vector

    :param args: For modes'CORNER' or 'CENTER' this has the form
        (width, height); for the 'RADIUS' this has the form
        (half_width, half_height); and for the 'CORNERS' mode, args
        should be the corner opposite to `coordinate`.

    :type: tuple

    :param mode: The drawing mode for the rectangle. Should be one of
        {'CORNER', 'CORNERS', 'CENTER', 'RADIUS'} (defaults to the
        mode being used by the sketch.)

    :type mode: str

    :returns: A rectangle.
    :rtype: p5.PShape

    """
    if mode is None:
        mode = _rect_mode

    if mode == 'CORNER':
        corner = coordinate
        width, height = args
    elif mode == 'CENTER':
        center = Point(*coordinate)
        width, height = args
        corner = Point(center.x - width/2, center.y - height/2, center.z)
    elif mode == 'RADIUS':
        center = Point(*coordinate)
        half_width, half_height = args
        corner = Point(center.x - half_width, center.y - half_height, center.z)
        width = 2 * half_width
        height = 2 * half_height
    elif mode == 'CORNERS':
        corner = Point(*coordinate)
        corner_2, = args
        corner_2 = Point(*corner_2)
        width = corner_2.x - corner.x
        height = corner_2.y - corner.y
    else:
        raise ValueError("Unknown rect mode {}".format(mode))

    return Rect(corner, width, height)

def square(coordinate, side_length, mode=None):
    """Return a square.

    :param coordinate: When mode is set to 'CORNER', the coordinate
        represents the lower-left corner of the square. For modes
        'CENTER' and 'RADIUS' the coordinate represents the center of
        the square.

    :type coordinate: tuple | list |p5.Vector

    :param side_length: The side_length of the square (for modes
        'CORNER' and 'CENTER') or hald of the side length (for the
        'RADIUS' mode)

    :type side_length: int or float

    :param mode: The drawing mode for the square. Should be one of
        {'CORNER', 'CORNERS', 'CENTER', 'RADIUS'} (defaults to the
        mode being used by the sketch.)

    :type mode: str

    :returns: A rectangle.
    :rtype: p5.PShape

    :raises ValueError: When the mode is set to 'CORNERS'

    """
    if mode is None:
        mode = _rect_mode

    if mode == 'CORNERS':
        raise ValueError("Cannot draw square with {} mode".format(mode))
    return rect(coordinate, side_length, side_length, mode=mode)

def rect_mode(mode='CORNER'):
    """Change the rect and square drawing mode for the sketch.

    :param mode: The new mode for drawing rects. Should be one of
        {'CORNER', 'CORNERS', 'CENTER', 'RADIUS'}. This defaults to
        'CORNER' so calling rect_mode without parameters will reset
        the sketch's rect mode.
    :type mode: str

    """
    global _rect_mode
    _rect_mode = mode

@_draw_on_return
def arc(coordinate, width, height, start_angle, stop_angle,
        mode='OPEN PIE', ellipse_mode=None):
    """Return a ellipse.

    :param coordinate: Represents the center of the arc when mode
        is 'CENTER' (the default) or 'RADIUS', the lower-left corner
        of the ellipse when mode is 'CORNER'.

    :rtype coordinate: 3-tuple

    :param width: For ellipse modes 'CORNER' or 'CENTER' this
        represents the width of the the ellipse of which the arc is a
        part. Represents the x-radius of the parent ellipse when
        ellipse mode is 'RADIUS

    :type width: float

    :param height: For ellipse modes 'CORNER' or 'CENTER' this
        represents the height of the the ellipse of which the arc is a
        part. Represents the y-radius of the parent ellipse when
        ellipse mode is 'RADIUS

    :type height: float

    :param mode: The mode used to draw an arc can be some combination
        of {'OPEN', 'CHORD', 'PIE'} separated by spaces. For instance,
        'OPEN PIE', etc (defaults to 'OPEN PIE')

    :type mode: str

    :param ellipse_mode: The drawing mode used for the ellipse. Should be one of
        {'CORNER', 'CENTER', 'RADIUS'} (defaults to the
        mode being used by the sketch.)

    :type mode: str

    :returns: An arc.
    :rtype: Arc

    """
    amode = mode

    if ellipse_mode is None:
        emode = _ellipse_mode
    else:
        emode = ellipse_mode

    if emode == 'CORNER':
        corner = Point(*coordinate)
        dim = Point(width, height)
        center = (corner.x + (dim.x / 2), corner.y + (dim.y / 2), corner.z)
    elif emode == 'CENTER':
        center = Point(*coordinate)
        dim = Point(width / 2, height / 2)
    elif emode == 'RADIUS':
        center = Point(*coordinate)
        dim = Point(width, height)
    else:
        raise ValueError("Unknown arc mode {}".format(emode))
    return Arc(center, dim, start_angle, stop_angle, attribs=amode)

@_draw_on_return
def ellipse(coordinate, *args, mode=None):
    """Return a ellipse.

    :param coordinate: Represents the center of the ellipse when mode
        is 'CENTER' (the default) or 'RADIUS', the lower-left corner
        of the ellipse when mode is 'CORNER' or, and an arbitrary
        corner when mode is 'CORNERS'.

    :type coordinate: 3-tuple

    :param args: For modes'CORNER' or 'CENTER' this has the form
        (width, height); for the 'RADIUS' this has the form
        (x_radius, y_radius); and for the 'CORNERS' mode, args
        should be the corner opposite to `coordinate`.

    :type: tuple

    :param mode: The drawing mode for the ellipse. Should be one of
        {'CORNER', 'CORNERS', 'CENTER', 'RADIUS'} (defaults to the
        mode being used by the sketch.)

    :type mode: str

    :returns: An ellipse
    :rtype: Arc

    """
    if mode is None:
        mode = _ellipse_mode

    if mode == 'CORNERS':
        corner = Point(*coordinate)
        corner_2, = args
        corner_2 = Point(*corner_2)
        width = corner_2.x - corner.x
        height = corner_2.y - corner.y
        mode = 'CORNER'
    else:
        width, height = args

    if mode == 'CORNER':
        corner = Point(*coordinate)
        dim = Point(width, height)
        center = (corner.x + (dim.x / 2), corner.y + (dim.y / 2), corner.z)
    elif mode == 'CENTER':
        center = Point(*coordinate)
        dim = Point(width / 2, height / 2)
    elif mode == 'RADIUS':
        center = Point(*coordinate)
        dim = Point(width, height)
    else:
        raise ValueError("Unknown ellipse mode {}".format(mode))

    return Ellipse(coordinate, width, height)

def circle(coordinate, radius, mode=None):
    """Return a circle.

    :param coordinate: Represents the center of the ellipse when mode
        is 'CENTER' (the default) or 'RADIUS', the lower-left corner
        of the ellipse when mode is 'CORNER' or, and an arbitrary
        corner when mode is 'CORNERS'.

    :type coordinate: 3-tuple

    :param radius: For modes'CORNER' or 'CENTER' this actually
        represents the diameter; for the 'RADIUS' this represents the
        radius.

    :type: tuple

    :param mode: The drawing mode for the ellipse. Should be one of
        {'CORNER', 'CORNERS', 'CENTER', 'RADIUS'} (defaults to the
        mode being used by the sketch.)

    :type mode: str

    :returns: A circle.
    :rtype: Ellipse

    :raises ValueError: When mode is set to 'CORNERS'

    """
    if mode is None:
        mode = _ellipse_mode

    if mode == 'CORNERS':
        raise ValueError("Cannot create circle in CORNERS mode")
    return ellipse(coordinate, radius, radius, mode=mode)

def ellipse_mode(mode='CENTER'):
    """Change the ellipse and circle drawing mode for the sketch.

    :param mode: The new mode for drawing ellipses. Should be one of
        {'CORNER', 'CORNERS', 'CENTER', 'RADIUS'}. This defaults to
        'CENTER' so calling ellipse_mode without parameters will reset
        the sketch's ellipse mode.
    :type mode: str

    """
    global _ellipse_mode
    _ellipse_mode = mode

def draw_shape(shape, pos=(0, 0, 0)):
    """Draw the given shape at the specified location.

    :param shape: The shape that needs to be drawn.
    :type shape: p5.PShape

    :param pos: Position of the shape
    :type pos: tuple | Vector

    """
    sketch.render(shape)
    for child_shape in shape.children:
        sketch.render(children)

def create_shape(kind=None, *args, **kwargs):
    """Create a new PShape

    Note :: A shape created using this function is *not* visible by
        default. Please make the shape visible by setting the shapes's
        `visible` attribute to true.

    :param kind: Type of shape. When left unspecified a generic PShape
        is returned (which can be edited later). Valid values for
        `kind` are: { 'point', 'line', 'triangle', 'quad', 'rect',
        'ellipse', 'arc', }

    :type kind: None | str

    :param args: Initial arguments to be passed to the shape creation
        function (only applied when `kind` is *not* None)

    :param kwargs: Initial keyword arguments to be passed to the shape
        creation function (only applied when `kind` is *not* None)

    :returns: The requested shape
    :rtype: p5.PShape

    """

    # TODO: add 'box', 'sphere' support
    valid_values = { None, 'point', 'line', 'triangle', 'quad',
                     'rect', 'square', 'ellipse', 'circle', 'arc', }

    def empty_shape(*args, **kwargs):
        return PShape()

    shape_map = {
        'arc': arc,
        'circle': circle,
        'ellipse': ellipse,
        'line': line,
        'point': point,
        'quad': quad,
        'rect': rect,
        'square': square,
        'triangle': triangle,
        None: empty_shape,
    }

    # kwargs['visible'] = False
    return shape_map[kind](*args, **kwargs)
