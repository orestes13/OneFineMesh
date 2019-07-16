'''
Constrained Delaunay Triangulations

David Bentley
27/03/2018
'''

import numpy as np
import shapely.geometry as sgeom
from shapely.ops import unary_union
from shapely.ops import split
from shapely.ops import triangulate as shapely_triangulate
from shapely.affinity import scale
from itertools import combinations, compress
import random

import matplotlib.pyplot as plt
from matplotlib import patches


# basic triangle class
class Triangle(sgeom.Polygon):
    def __init__(self, points, ghost=False):
        '''
        Class to overload the sgeom.Polygon class, to allow additional
        attributes for use in the Delaunay triangulation.

        Arguments:
            points: an array containing the vertices of the triangle

        Keyword arguments:
            ghost: a boolean describing whether the triangle is a ghost
                   triangle or not
        '''
        # ensure the points are counterclockwise
        if len(points) != 3:
            raise ValueError('must provide 3 points')
        centre = sum(np.array(p) for p in points) / len(points)
        theta = []
        for p in points:
            vec = np.array(p) - centre
            theta.append((np.arctan2(vec[1], vec[0]) + 2 * np.pi) % (2 * np.pi))
        idx = np.argsort(theta)

        super().__init__(np.array(points)[idx])
        self.vertices = sgeom.MultiPoint(points)
        self.is_ghost = ghost
        self.centre = self.centroid.coords[0]

        # circumcircle
        if not self.is_ghost:
            cc, cr = circumcircle(self)
            self.circumcentre = cc
            self.circumradius = cr

        # calculate the interior angles
        self.angles = []
        for i, c in enumerate(self.vertices):
            c_m = np.array(self.vertices[i - 1])
            c_p = np.array(self.vertices[(i + 1) % len(self.vertices)])
            vec_m = np.array(c) - c_m
            vec_p = np.array(c) - c_p
            num = np.dot(vec_m, vec_p)
            denom = np.linalg.norm(vec_m) * np.linalg.norm(vec_p)
            self.angles.append(np.arccos(num / denom))


# 2D triangulations
class DelaunayTriangulation(object):
    def __init__(self, shape):
        '''
        Construct a Delaunay triangulation of the provided shape

        Arguments:
            shape: a shapely.geometry defining the shape to be triangulated
        '''
        self.shape = shape

        if isinstance(shape, sgeom.Polygon):
            points = list(shape.exterior.coords)
            for interior in shape.interiors:
                points.extend(interior.coords)
        elif isinstance(shape, sgeom.MultiLineString):
            points = []
            for line in shape:
                points.extend(line.coords)
            points = list(set(points))
        elif isinstance(shape, sgeom.MultiPoint):
            points = shape
        else:
            msg = 'given shape is of type {}, require MultiPoint, MultiLineString, Polygon or MultiPolygon'.format(type(shape))
            raise TypeError(msg)
        self.points = sgeom.MultiPoint(points)

        # determine the edges of the shape
        edges = []
        if isinstance(shape, sgeom.MultiPoint):
            self.segments = None
        elif isinstance(shape, sgeom.MultiLineString):
            self.segments = shape
        else:
            # Polygon
            self.segments = []
            exterior = sgeom.LineString(shape.exterior)
            for p1, p2 in zip(exterior.coords[:-1], exterior.coords[1:]):
                self.segments.append(sgeom.LineString((p1, p2)))
            for interior in shape.interiors:
                for p1, p2 in zip(interior.coords[:-1], interior.coords[1:]):
                    self.segments.append(sgeom.LineString((p1, p2)))
            self.segments = sgeom.MultiLineString(self.segments)

        # calculate the Delaunay triangulation
        self.triangles = triangulate(self.points)
        self.real_triangles = [tri for tri in self.triangles if not tri.is_ghost]

    # method for adding additional points, as required for refinement
    def insert_point(self, point):
        '''
        Method to insert a point into the Delaunay triangulation, and
        recompute the triangulation.

        Arguments:
            point: the point to be inserted into the triangulation
        '''
        self.points, triangles = insert_point(point, self.points,
                                              self.triangles)
        # exclude ghost triangles
        self.triangles = triangles
        self.real_triangles = [tri for tri in triangles if not tri.is_ghost]


class ConstrainedDelaunayTriangulation(DelaunayTriangulation):
    def __init__(self, shape):
        '''
        Construct a constrained Delaunay triangulation of the provided
        shape

        Arguments:
            shape: a shapely.geometry defining the shape to be triangulated
        '''
        # construct the Delaunay triangulation
        super().__init__(shape)

        # determine which of the triangles intersect the edges
        self.triangles = np.array(self.triangles)
        for edge in self.segments:
            triangles_to_remove = np.zeros(len(self.triangles), dtype=bool)
            for idx, tri in enumerate(self.triangles):
                if edge.intersects(tri.boundary):
                    if isinstance(edge.intersection(tri.boundary), sgeom.MultiPoint):
                        triangles_to_remove[idx] = True
            bad_triangles = self.triangles[triangles_to_remove]
            self.triangles = self.triangles[~triangles_to_remove]

            if len(bad_triangles) > 0:
                new_triangles = insert_segment(edge, list(bad_triangles))
                self.triangles = list(self.triangles)
                self.triangles.extend(new_triangles)
                self.triangles = np.array(self.triangles)

        # remove any triangles not in the provided shape
        self.triangles = [tri for tri in self.triangles if shape.contains(tri)]

    # method for adding additional points, as required for refinement
    def insert_point(self, point):
        '''
        Method to insert a point into the constrained Delaunay triangulation,
        and recompute the triangulation.

        Arguments:
            point: the point to be inserted into the triangulation
        '''
        self.points, triangles = insert_point(point, self.points,
                                              self.triangles)

        self.triangles = [tri for tri in triangles if self.shape.buffer(1e-10).contains(tri)]


class ConformalDelaunayTriangulation(ConstrainedDelaunayTriangulation):
    def __init__(self, shape):
        '''
        Construct a conformal Delaunay triangulation of the provided
        shape

        Arguments:
            shape: a shapely.geometry defining the shape to be triangulated
        '''
        super().__init__(shape)

        # determine which triangles are not Delaunay
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        for p in self.points:
            plt.plot(*p.buffer(0.01).exterior.xy, color='black')
        for tri in self.triangles:
            cc, cr = circumcircle(tri)
            if any([point_within_circumcircle(p, cc, cr) for p in self.points.difference(tri.vertices)]):
                patch = patches.Polygon(np.array(tri.exterior.coords.xy).T,
                                        facecolor='red', edgecolor='black')
                plt.plot(*cc.buffer(cr, resolution=1024).exterior.xy,
                         linestyle='--', color='gray')
            else:
                patch = patches.Polygon(np.array(tri.exterior.coords.xy).T,
                                        facecolor='blue', edgecolor='black')
            ax.add_patch(patch)
        plt.show()


# generic functions
def insert_point(point, vertices, triangles):
    '''
    Insert a point into a triangulation, recalculating the triangulation as
    appropriate.

    Arguments:
        point: the point to be inserted
        vertices: the vertices of the triangulation
        triangles: the triangles of the triangulation to be recalculated
    '''
    triangles_to_remove = np.zeros(len(triangles), dtype=bool)
    for idx, tri in enumerate(triangles):
        ghost_vertices = []
        tri_points = tri.vertices
        if tri.is_ghost:
            # ghost triangle
            ghost_vertex = tri_points.difference(vertices)
            ghost_normal, edge = ghost_circumcircle(tri, ghost_vertex)
            mid_point = edge.interpolate(0.5, normalized=True)
            line = sgeom.LineString([mid_point, point])
            vec1 = np.squeeze(np.diff(np.array(ghost_normal), axis=0))
            vec2 = np.squeeze(np.diff(np.array(line), axis=0))
            cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            if sgeom.Point(point).intersects(edge) or cos_angle > 0:
                triangles_to_remove[idx] = True
                ghost_vertices.append(ghost_vertex)
        else:
            # normal triangle
            if point_within_circumcircle(point, tri.circumcentre,
                                         tri.circumradius):
                triangles_to_remove[idx] = True

    # construct the union of the containing triangles, and remove
    # from the triangulation
    good_triangles = list(np.array(triangles)[~triangles_to_remove])
    bad_triangles = list(np.array(triangles)[triangles_to_remove])
    containing_union = unary_union(bad_triangles)

    # remove the containing shape, and construct new triangles
    vertices = vertices.union(point)

    # associate each segment of the containing union with a bad triangle
    if not isinstance(containing_union, (sgeom.Polygon,
                                         sgeom.MultiPolygon)):
        msg = 'expected the cavity to be of type Polygon or MultiPolygon, got {}'.format(type(containing_union))
        raise TypeError(msg)

    # remove ghost triangles
    if isinstance(containing_union, sgeom.Polygon):
        for i, x in enumerate(containing_union.boundary.coords[:-1]):
            edge = sgeom.LineString([x, containing_union.boundary.coords[i +  1]])
            if point.buffer(1e-9).intersects(edge):
                continue
            else:
                new_tri = Triangle([x, containing_union.boundary.coords[i +  1],
                                    np.array(point)])
                tri_points = new_tri.vertices
                real_points = vertices.intersection(tri_points)
                if len(real_points) == 3:
                    good_triangles.append(new_tri)
    elif isinstance(containing_union, sgeom.MultiPolygon):
        for pol in containing_union:
            for i, x in enumerate(pol.boundary.coords[:-1]):
                edge = sgeom.LineString([x, pol.boundary.coords[i +  1]])
                if point.buffer(1e-9).intersects(edge):
                    continue
                else:
                    new_tri = Triangle([x, pol.boundary.coords[i +  1],
                                        np.array(point)])
                    tri_points = new_tri.vertices
                    real_points = vertices.intersection(tri_points)
                    if len(real_points) == 3:
                        good_triangles.append(new_tri)

    # recompute ghost triangles
    ghosts = sgeom.MultiPolygon(ghost_triangles(vertices))
    for t in good_triangles:
        if not t.is_valid:
            print('invalid triangle', t)
    new_ghosts = ghosts.difference(unary_union(good_triangles))
    if isinstance(new_ghosts, sgeom.Polygon):
        good_triangles.append(Triangle(new_ghosts.exterior.coords[:-1], ghost=True))
    elif isinstance(new_ghosts, sgeom.MultiPolygon):
        good_triangles.extend([Triangle(g.exterior.coords[:-1], ghost=True) for g in new_ghosts])

    triangles = good_triangles
    return vertices, triangles


def insert_segment(segment, containing_triangles):
    '''
    Re-triangulate a polygon

    Arguments:
        segment: the segment to be inserted
        containing_triangles: the collection of triangles intersected by the
                              segment to be inserted
    '''
    containing_union = unary_union(containing_triangles)
    new_triangles = []

    # make a list of edges intersecting the segment to be inserted
    intersecting_edges = []
    for (tri1, tri2) in combinations(containing_triangles, 2):
        intersection = tri1.intersection(tri2)
        if isinstance(intersection, sgeom.LineString):
            intersecting_edges.append(intersection)

    # for each edge, identify the pair of triangles sharing that edge,
    # and swap the vertices connected to form two new triangles, and
    # a new edge
    new_edges = []
    while len(intersecting_edges) > 0:
        edge = random.choice(intersecting_edges)
        intersecting_edges.remove(edge)
        connected_triangles = []
        for tri in containing_triangles:
            if edge.intersects(tri):
                if isinstance(edge.intersection(tri.boundary), sgeom.LineString):
                    connected_triangles.append(tri)
        tri1, tri2 = connected_triangles
        union_polygon = unary_union(connected_triangles)
        if not union_polygon.equals(union_polygon.convex_hull):
            intersecting_edges.insert(0, edge)
            continue
        for tri in connected_triangles:
            containing_triangles.remove(tri)
        tri1_coords = tri1.vertices
        tri2_coords = tri2.vertices
        connected_vertices = tri1_coords.intersection(tri2_coords)
        unconnected_vertices = tri1_coords.symmetric_difference(tri2_coords)
        new_edge = sgeom.LineString(list(unconnected_vertices))
        flipped_triangles = []
        for v in connected_vertices:
            new_tri = Triangle(unconnected_vertices.union(v))
            flipped_triangles.append(new_tri)
        containing_triangles.extend(flipped_triangles)
        # check if the new edge intersects the segment (somewhere other than at
        # the end of the segment))
        if segment.intersects(new_edge):
            distance = np.array([segment.intersection(new_edge).distance(sgeom.Point(p)) for p in segment.coords])
            if all(distance > 1e-12):
                # new edge intersects the segment, so do not keep
                intersecting_edges.append(new_edge)
                #containing_triangles.extend(flipped_triangles)
                continue

        new_edges.append(new_edge)

    new_triangles = []
    for edge in new_edges:
        for tri in containing_triangles:
            if edge.intersects(tri):
                if isinstance(edge.intersection(tri.boundary), sgeom.LineString):
                    if not any([tri.equals(t) for t in new_triangles]):
                        new_triangles.append(tri)

    # now check that each of the new edges are locally Delaunay
    for edge in new_edges:
        if edge.equals(segment):
            continue
        connected_triangles = []
        for tri in new_triangles:
            if edge.intersects(tri):
                if isinstance(edge.intersection(tri.boundary), sgeom.LineString):
                    connected_triangles.append(tri)
        tri1, tri2 = connected_triangles
        cc1, cr1 = circumcircle(tri1)
        cc2, cr2 = circumcircle(tri2)
        if any([point_within_circumcircle(p, cc1, cr1) for p in tri2.boundary.coords]):
            # not locally Delaunay, so flip triangles
            tri1_coords = tri1.vertices
            tri2_coords = tri2.vertices
            connected_vertices = tri1_coords.intersection(tri2_coords)
            unconnected_vertices = tri1_coords.symmetric_difference(tri2_coords)
            flipped_triangles = []
            for v in connected_vertices:
                new_tri = Triangle(unconnected_vertices.union(v))
                flipped_triangles.append(new_tri)
            t1, t2 = flipped_triangles
            new_triangles.remove(tri1)
            new_triangles.remove(tri2)
            new_triangles.extend(flipped_triangles)

    return new_triangles


def point_within_circumcircle(point, circumcentre, circumradius):
    '''
    Determine whether the given point is within the circumcircle defined
    by the given circumcentre and circumradius.

    Arguments:
        point: the point to check
        circumcenrte: the centre of the circumcircle
        circumradius: the radius of the circumcircle
    '''
    connecting_vector = np.array(point) - np.array(circumcentre)
    distance = np.linalg.norm(connecting_vector)
    return distance <= (1 - 1e-9) * circumradius


#@profile
def circumcircle(triangle):
    '''
    Given a triangle, construct the circumcircle

    Arguments:
        triangle: a Triangle defining the triangle
                  for which to find the circumcircle.
    '''
    # extract the vertices of the triangle
    p1, p2, p3 = triangle.vertices

    # calculate the circumcentre and circumradius
    a = np.linalg.det(np.array([[p1.x, p1.y, 1],
                                [p2.x, p2.y, 1],
                                [p3.x, p3.y, 1]]))
    b_x = - np.linalg.det(np.array([[np.dot(p1, p1), p1.y, 1],
                                    [np.dot(p2, p2), p2.y, 1],
                                    [np.dot(p3, p3), p3.y, 1]]))
    b_y = np.linalg.det(np.array([[np.dot(p1, p1), p1.x, 1],
                                  [np.dot(p2, p2), p2.x, 1],
                                  [np.dot(p3, p3), p3.x, 1]]))
    c = - np.linalg.det(np.array([[np.dot(p1, p1), p1.x, p1.y],
                                  [np.dot(p2, p2), p2.x, p2.y],
                                  [np.dot(p3, p3), p3.x, p3.y]]))
    x_c = -b_x / (2 * a)
    y_c = -b_y / (2 * a)
    circumcentre = sgeom.Point((x_c, y_c))
    circumradius = np.sqrt(b_x**2 + b_y**2 - 4 * a * c) / (2 * np.abs(a))

    return (circumcentre, circumradius)

def ghost_circumcircle(triangle, ghost_vertex):
    '''
    Given a ghost triangle, return the circumcircle. For a ghost triangle,
    the circumcircle is actually the open half plane tangential to the edge
    of the triangle of the ghost vertex, and containing the ghost vertex
    (and edge).

    Since we can not define an open half plane, we instead compute the (unit)
    normal to the edge opposite the ghost vertex, oriented towards the
    ghost vertex. A point is then considered to be in the circumcircle of
    the ghost triangle if the angle with the unit normal is in (-90, 90),
    or if the point lies on the edge.

    Arguments:
        triangle: a shapely.geometry.LineString defining the triangle
                  for which to find the circumcircle.
        ghost_vertex: a shapely.geometry.Point defining the ghost vertex
    '''
    # find the edge opposite the ghost vertex
    ball = ghost_vertex.buffer(0.01)
    vertices = list(zip(*triangle.exterior.xy))
    edges = []
    for i in range(len(vertices) - 1):
        edges.append(sgeom.LineString([vertices[i], vertices[i + 1]]))
    idx = np.squeeze(np.where([not ball.intersects(edge) for edge in edges]))
    #print(idx)
    opposite_edge = edges[idx]

    # construct the normal to the edge, oriented towards the ghost vertex
    mid_point = opposite_edge.interpolate(0.5, normalized = True)
    normal = sgeom.LineString([mid_point, ghost_vertex])
    scale = 1.0 / normal.length
    normal = sgeom.LineString([mid_point, (mid_point.x + scale * (ghost_vertex.x - mid_point.x),
                                           mid_point.y + scale * (ghost_vertex.y - mid_point.y))])
    return normal, opposite_edge

#@profile
def perpendicular_line(line, half_length=0.5):
    '''
    Given a line, calculate the equation of the perpendicular line.

    Arguments:
        line: a shapely.geometry.LineString defining the line

    Keyword arguments:
        half_length: half the desired length of the calculated perpendicular
                     line
    '''
    # slope of line
    e1, e2 = line.boundary
    try:
        slope = float(e2.y - e1.y) / (e2.x - e1.x)
    except ZeroDivisionError:
        slope = np.inf

    # midpoint of line
    mid_point = np.array(line.interpolate(0.5, normalized = True))

    # slope of perpendicular line
    if slope == np.inf:
        perp_slope = 0.0
    else:
        try:
            perp_slope = - 1.0 / slope
        except ZeroDivisionError:
            perp_slope = np.inf

    # determine the end points of the perpendicular line
    if perp_slope == np.inf:
        p1 = mid_point - np.array([0, half_length])
        p2 = mid_point + np.array([0, half_length])
    else:
        theta = np.arctan2(perp_slope, 1.0)
        x_i = half_length * np.cos(theta)
        y_i = half_length * np.sin(theta)
        p1 = mid_point - np.array([x_i, y_i])
        p2 = mid_point + np.array([x_i, y_i])

    return sgeom.LineString([p1, p2])


def ghost_triangles(points, infinity=100.0):
    '''
    Given a set of points, calculate a set of ghost triangles, each of
    which shares an edge with the convex hull

    Arguments:
        points: a shapely.geometry.MultiPoint defining the points to be
                triangulated

    Keyword arguments:
        infinity: a large distance
    '''
    # calculate the edges of the convex hull
    convex_hull = points.convex_hull
    boundary = convex_hull.boundary
    if not isinstance(boundary, sgeom.LineString):
        print('convex hull is type {}'.format(type(boundary)))
    # workaround for co-linear points, whice are missing from the convex hull
    index = [p.intersects(boundary) for p in points]
    points_on_boundary = np.array(points)[index]
    if len(boundary.coords) <= len(points_on_boundary):
        # same length since polygon is closed, so one point is repeated
        centre = convex_hull.centroid
        # sort into (clockwise) order
        shifted_points = points_on_boundary - np.array(centre)
        angles = np.arctan2(shifted_points[:, 1], shifted_points[:, 0])
        points_on_boundary = points_on_boundary[np.argsort(angles)]
        boundary = sgeom.Polygon(points_on_boundary).boundary
    boundary_lines = []
    for p1, p2 in zip(boundary.coords[:-1], boundary.coords[1:]):
        boundary_lines.append(sgeom.LineString([p1, p2]))

    # calculate the ghost triangles
    ghost_triangles = []
    for line in boundary_lines:
        v1, v2 = line.coords
        midpoint = line.interpolate(0.5, normalized=True)
        normal = perpendicular_line(line, half_length=infinity)
        for i, c in enumerate(normal.coords):
            normal_segment = sgeom.LineString([midpoint, c])
            if isinstance(normal_segment.intersection(convex_hull), sgeom.LineString):
                if normal_segment.intersection(convex_hull).length < 1e-12:
                    ghost_triangles.append(Triangle([c, v1, v2], ghost=True))
            else:
                ghost_triangles.append(Triangle([c, v1, v2], ghost=True))

    return ghost_triangles

#@profile
def triangulate(points):
    '''
    Construct the Delaunay triangulation of a set of points

    Arguments:
        points: a shapely.geometry.MultiPoint defining the points to be
                triangulated
    '''
    triangles = []

    # choose three random points in the point set to form an initial triangle
    inserted_points = sgeom.MultiPoint(random.sample(list(points), 3))
    points = points.difference(inserted_points)
    initial_triangle = Triangle([(p.x, p.y) for p in inserted_points])
    triangles.append(initial_triangle)

    # calculate the ghost triangles
    triangles.extend(ghost_triangles(inserted_points))

    # loop through the remaining points in a random order, inserting them
    # into the bounding triangle and constructing the triangulation
    while len(points) > 0:
        p = random.choice(points)
        try:
            points = sgeom.MultiPoint(points.difference(p))
        except TypeError:
            points = sgeom.MultiPoint([points.difference(p)])
        inserted_points, triangles = insert_point(p, inserted_points, triangles)

    # reset points
    points = inserted_points

    return triangles#[tri for tri in triangles if not tri.is_ghost]


if __name__ == "__main__":
    n = 100
    points = sgeom.MultiPoint([np.random.random(2) for _ in range(n)])

    filename = "/home/dbentley/code/OneFineMesh/onefinemesh/tests/ne_10m_lakes.shp"

    import fiona
    import shapely
    f = fiona.open(filename)
    lake_michigan = sgeom.shape([pol for pol in f if pol['properties']['name'] == 'Lake Michigan'][0]['geometry'])
    lake_superior = sgeom.shape([pol for pol in f if pol['properties']['name'] == 'Lake Superior'][0]['geometry'])

    lines = []
    n = 30
    for _ in range(n):
        r = 0.5 + np.random.random()
        theta = 2 * np.pi * _ / n
        lines.append(sgeom.LineString([(0, 0), r * np.array([np.cos(theta), np.sin(theta)])]))
    lines.append(sgeom.LineString([(-2, -2), (-2, 2)]))
    lines.append(sgeom.LineString([(-2, 2), (2, 2)]))
    lines.append(sgeom.LineString([(2, 2), (2, -2)]))
    lines.append(sgeom.LineString([(2, -2), (-2, -2)]))
    star = sgeom.MultiLineString(lines)

    import sys
    sys.path.append('../../tests')
    import alphabet

#    import string
#    for letter in string.ascii_lowercase:
    c = alphabet.letter_to_polygon('k')

#    dt = DelaunayTriangulation(points)
#    dt = ConformalDelaunayTriangulation(lake_michigan)
#    dt = ConstrainedDelaunayTriangulation(lake_superior)
#    dt = ConstrainedDelaunayTriangulation(star)
    dt = ConstrainedDelaunayTriangulation(c)

    from refinements import ruppert

    ruppert(dt, alpha=30)

    fig = plt.figure()
    fig.add_subplot(1, 2, 1)
    plt.plot(*c.exterior.xy, color='blue', linewidth=2)
    for interior in c.interiors:
        plt.plot(*interior.xy, color='blue', linewidth=2)
    for t in dt.triangles:
        plt.plot(*t.exterior.xy, linewidth=1, color='black')
    #for e in dt.segments:
    #    plt.plot(*e.xy, linewidth=2, color='blue')
    for p in dt.points:
        plt.plot(*p.buffer(0.01).exterior.xy, color='red', alpha=0.5)
    fig.add_subplot(1, 2, 2)
    triangles = shapely_triangulate(dt.points)
    for t in triangles:
        plt.plot(*t.exterior.xy, linewidth=1, color='black')
    for p in dt.points:
        plt.plot(*p.buffer(0.01).exterior.xy, color='red', alpha=0.5)
    plt.show()
