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

# 2D triangulations
class DelaunayTriangulation(object):
    def __init__(self, shape):
        '''
        Construct a Delaunay triangulation of the provided shape

        Arguments:
            shape: a shapely.geometry defining the shape to be triangulated
        '''
        if isinstance(shape, sgeom.Polygon):
            points = list(shape.exterior.coords)
            for interior in shape.interiors:
                points.extend(interior.coords)
        elif isinstance(shape, sgeom.MultiLineString):
            points = []
            for line in shape:
                points.extend(line.coords)
            points = list(set(points))
        else:
            msg = 'given shape is of type {}, require MultiPoint, MultiLineString, Polygon or MultiPolygon'.format(type(shape))
            raise TypeError(msg)
        self.points = sgeom.MultiPoint(points)
        self.triangles = triangulate(self.points)

class ConstrainedDelaunayTriangulation(DelaunayTriangulation):
    def __init__(self, shape):
        '''
        Construct a constrained Delaunay triangulation of the provided
        shape

        Arguments:
            shape: a shapely.geometry defining the shape to be triangulated
        '''
        # construct the Delaunay triangulation
        super(ConstrainedDelaunayTriangulation, self).__init__(shape)

        # determine the edges of the shape
        edges = []
        if isinstance(shape, sgeom.MultiPoint):
            msg = '{} has no edges to constrain the triangulation'.format(type(shape))
            raise TypeError(msg)
        elif isinstance(shape, sgeom.MultiLineString):
            self.edges = shape
        else:
            # Polygon
            self.edges = []
            exterior = sgeom.LineString(shape.exterior)
            for p1, p2 in zip(exterior.coords[:-1], exterior.coords[1:]):
                self.edges.append(sgeom.LineString((p1, p2)))
            for interior in shape.interiors:
                for p1, p2 in zip(interior.coords[:-1], interior.coords[1:]):
                    self.edges.append(sgeom.LineString((p1, p2)))
            self.edges = sgeom.MultiLineString(self.edges)

        # determine which of the triangles intersect the edges
        self.triangles = np.array(self.triangles)
        for edge in self.edges:
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
        #idx = [shape.contains(tri) for tri in self.triangles]
        #self.triangles = list(self.triangles[idx])


class ConformalDelaunayTriangulation(ConstrainedDelaunayTriangulation):
    def __init__(self, shape):
        '''
        Construct a conformal Delaunay triangulation of the provided
        shape

        Arguments:
            shape: a shapely.geometry defining the shape to be triangulated
        '''
        super(ConformalDelaunayTriangulation, self).__init__(shape)

# 3D tetrahedralisations
#class DelaunayTetrahedralisation(object)

# generic functions
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
        edge = intersecting_edges.pop()
        connected_triangles = []
        for tri in containing_triangles:
            if edge.intersects(tri):
                if isinstance(edge.intersection(tri.boundary), sgeom.LineString):
                    connected_triangles.append(tri)
        tri1, tri2 = connected_triangles
        union_polygon = unary_union(connected_triangles)
        lines = []
        for p1, p2 in zip(union_polygon.boundary.coords[:-1], union_polygon.boundary.coords[1:]):
            lines.append(sgeom.LineString([p1, p2]))
        lines.append(lines[0])
        angles = []
        for i, line in enumerate(lines[:-1]):
            line2 = lines[i + 1]
            v1 = np.array(line)[0] - np.array(line)[1]
            v2 = np.array(line2)[0] - np.array(line2)[1]
            theta = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
            angles.append(theta)
        if any(np.array(angles) < 0):
            print(angles)
        for tri in connected_triangles:
            containing_triangles.remove(tri)
        tri1_coords = set(tri1.boundary.coords)
        tri2_coords = set(tri2.boundary.coords)
        connected_vertices = tri1_coords.intersection(tri2_coords)
        unconnected_vertices = tri1_coords.symmetric_difference(tri2_coords)
        new_edge = sgeom.LineString(list(unconnected_vertices))
        flipped_triangles = []
        for v in connected_vertices:
            new_tri = sgeom.Polygon(tuple(unconnected_vertices) + (v,))
            flipped_triangles.append(new_tri)
        containing_triangles.extend(flipped_triangles)
        # check if the new edge intersect the segment (somewhere other than at
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
        c1 = circumcircle(tri1, None)
        c2 = circumcircle(tri2, None)
        if any([c1.contains(sgeom.Point(p)) for p in tri2.boundary.coords]):
            # not locally Delaunay, so flip triangles
            tri1_coords = set(tri1.boundary.coords)
            tri2_coords = set(tri2.boundary.coords)
            connected_vertices = tri1_coords.intersection(tri2_coords)
            unconnected_vertices = tri1_coords.symmetric_difference(tri2_coords)
            flipped_triangles = []
            for v in connected_vertices:
                new_tri = sgeom.Polygon(tuple(unconnected_vertices) + (v,))
                flipped_triangles.append(new_tri)
            t1, t2 = flipped_triangles
            new_triangles.remove(tri1)
            new_triangles.remove(tri2)
            new_triangles.extend(flipped_triangles)

    return new_triangles


def circumcircle(triangle, resolution=16):
    '''
    Given a triangle, construct the circumcircle

    Arguments:
        triangle: a shapely.geometry.LineString defining the triangle
                  for which to find the circumcircle.
    '''
    # determine the longest edge of the triangle
    edges = []
    for i, p in enumerate(triangle.boundary.coords[:-1]):
        p2 = triangle.boundary.coords[i + 1]
        edges.append(sgeom.LineString([p, p2]))
    longest_edge = max([l.length for l in edges])

    # compute the perpendicalar bisectors of two of the edges
    pb1 = perpendicular_line(edges[0], longest_edge)
    pb2 = perpendicular_line(edges[1], longest_edge)

    # find the point of intersection of these two bisectors, i.e., the
    # circumcentre
    circumcentre = pb1.intersection(pb2)
    while circumcentre.is_empty:
        pb1 = scale(pb1, xfact=2.0, yfact=2.0)
        pb2 = scale(pb2, xfact=2.0, yfact=2.0)
        circumcentre = pb1.intersection(pb2)

    # determine the circumradius
    line = sgeom.LineString([circumcentre, triangle.boundary.coords[0]])
    circumradius = line.length

    # determine the resolution
    if resolution is None:
        # calculate shortest side of triangle
        sides = []
        for i, x in enumerate(triangle.boundary.coords[:-1]):
            sides.append(sgeom.LineString([x, triangle.boundary.coords[i +  1]]).length)
        s = min(sides)
        n_segments = 2 * np.pi * circumradius / s
        # ensure power of 2
        n_segments = int(np.power(2, np.ceil(np.log2(n_segments))))
        resolution = max(n_segments, 16)  # 16 segments minimum

    return circumcentre.buffer(circumradius, resolution)

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
    opposite_edge = edges[idx]

    # construct the normal to the edge, oriented towards the ghost vertex
    mid_point = opposite_edge.interpolate(0.5, normalized = True)
    normal = sgeom.LineString([mid_point, ghost_vertex])
    scale = 1.0 / normal.length
    normal = sgeom.LineString([mid_point, (mid_point.x + scale * (ghost_vertex.x - mid_point.x),
                                           mid_point.y + scale * (ghost_vertex.y - mid_point.y))])
    return normal, opposite_edge


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
    x2, y2 = line.coords[-1]
    x1, y1 = line.coords[0]
    try:
        slope = float(y2 - y1) / (x2 - x1)
    except ZeroDivisionError:
        slope = np.inf

    # midpoint of line
    mid_point = line.interpolate(0.5, normalized = True)

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
        p1 = sgeom.Point(np.array(mid_point) - np.array([0, half_length]))
        p2 = sgeom.Point(np.array(mid_point) + np.array([0, half_length]))
    else:
        theta = np.arctan2(perp_slope, 1.0)
        x_i = half_length * np.cos(theta)
        y_i = half_length * np.sin(theta)
        p1 = sgeom.Point(np.array(mid_point) - np.array([x_i, y_i]))
        p2 = sgeom.Point(np.array(mid_point) + np.array([x_i, y_i]))

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
    boundary_lines = []
    for i in range(len(boundary.coords) - 1):
        p1, p2 = np.array(boundary.coords)[i:i+2]
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
                    ghost_triangles.append(sgeom.Polygon([c, v1, v2]))
            else:
                ghost_triangles.append(sgeom.Polygon([c, v1, v2]))

    if not len(boundary_lines) == len(ghost_triangles):
        plt.plot(*convex_hull.exterior.xy, linewidth=3, color='black')
        for tri in ghost_triangles:
            plt.plot(*tri.exterior.xy, linewidth=2, color='blue')
        plt.show()

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
    cache = {}

    # choose three random points in the point set to form an initial triangle
    inserted_points = sgeom.MultiPoint(random.sample(list(points), 3))
    points = points.difference(inserted_points)
    initial_triangle = sgeom.Polygon([(p.x, p.y) for p in inserted_points])
    triangles.append(initial_triangle)

    # calculate the ghost triangles
    triangles.extend(ghost_triangles(inserted_points))

    # loop through the remaining points in a random order, inserting them
    # into the bounding triangle and constructing the triangulation
    while len(points) > 0:
        p = random.choice(points)
        print(len(points))
        try:
            points = sgeom.MultiPoint(points.difference(p))
        except TypeError:
            points = sgeom.MultiPoint([points.difference(p)])

        # determine if the new point is inside the point set
        convex_hull = inserted_points.convex_hull
        interior_point = p.within(convex_hull)

        triangles_to_remove = np.zeros(len(triangles), dtype=bool)
        for idx, tri in enumerate(triangles):
            ghost_vertices = []
            #plt.plot(*tri.exterior.xy)
            #plt.show()
            tri_points = sgeom.MultiPoint(list(set(list(tri.exterior.coords))))
            real_points = inserted_points.intersection(tri_points)
            centroid = tuple(tri.centroid.coords)[0]
            if len(real_points) == 3:
                # normal triangle
                try:
                    c = cache[centroid]
                except KeyError:
                    c = circumcircle(tri, None)
                    cache[centroid] = c
                if p.within(c):
                    triangles_to_remove[idx] = True
            elif len(real_points) == 2:
                # ghost triangle
                ghost_vertex = tri_points.difference(inserted_points)
                #print(tri_points, inserted_points)
                ghost_normal, edge = ghost_circumcircle(tri, ghost_vertex)
                mid_point = edge.interpolate(0.5, normalized=True)
                line = sgeom.LineString([mid_point, p])
                vec1 = np.squeeze(np.diff(np.array(ghost_normal), axis=0))
                vec2 = np.squeeze(np.diff(np.array(line), axis=0))
                cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                #print('cos angle', cos_angle)
                #if sgeom.Point(ghost_vertex).intersects(edge):
                #    print('inserted point on edge')
                if cos_angle > 0:
                    triangles_to_remove[idx] = True
                    ghost_vertices.append(ghost_vertex)
            else:
                raise ValueError('unable to determine if triangle is real or a ghost')

        # construct the union of the containing triangles, and remove
        # from the triangulation
        good_triangles = list(np.array(triangles)[~triangles_to_remove])
        bad_triangles = list(np.array(triangles)[triangles_to_remove])
        for tri in bad_triangles:
            try:
                del cache[tuple(tri.centroid.coords)[0]]
            except KeyError:
                pass
        containing_union = unary_union(bad_triangles)

#        fig = plt.figure()
#        fig.add_subplot(1, 1, 1)
#        plt.plot(*p.buffer(0.01).exterior.xy, color='red')
#        for tri in good_triangles:
#            plt.plot(*tri.exterior.xy, color='black')
#        for tri in bad_triangles:
#            plt.plot(*tri.exterior.xy, color='blue')
#        plt.show()

        # remove the containing shape, and construct new triangles
        inserted_points = inserted_points.union(p)
        # remove ghost triangles
        if isinstance(containing_union, sgeom.Polygon):
            if isinstance(containing_union.boundary, sgeom.MultiLineString):
                for line in containing_union.boundary:
                    plt.plot(*line.xy)
                    print(line)
                plt.show()
            for i, x in enumerate(containing_union.boundary.coords[:-1]):
                new_tri = sgeom.Polygon([x, containing_union.boundary.coords[i +  1], np.array(p)])
                tri_points = sgeom.MultiPoint(list(set(list(new_tri.exterior.coords))))
                real_points = inserted_points.intersection(tri_points)
                if len(real_points) == 3:
                    good_triangles.append(new_tri)
        elif isinstance(containing_union, sgeom.MultiPolygon):
            for pol in containing_union:
                if isinstance(pol.boundary, sgeom.MultiLineString):
                    for line in pol.boundary:
                        plt.plot(*line.xy)
                        print(line)
                    plt.show()
                for i, x in enumerate(pol.boundary.coords[:-1]):
                    new_tri = sgeom.Polygon([x, pol.boundary.coords[i +  1], np.array(p)])
                    tri_points = sgeom.MultiPoint(list(set(list(new_tri.exterior.coords))))
                    real_points = inserted_points.intersection(tri_points)
                    if len(real_points) == 3:
                        good_triangles.append(new_tri)

        # recompute ghost triangles
        ghosts = sgeom.MultiPolygon(ghost_triangles(inserted_points))
        new_ghosts = ghosts.difference(unary_union(good_triangles))
        good_triangles.extend(list(new_ghosts))

        triangles = good_triangles

    # reset points
    points = inserted_points

    interior_triangles = []
    for tri in triangles:
        tri_points = sgeom.MultiPoint(list(set(list(tri.exterior.coords))))
        real_points = inserted_points.intersection(tri_points)
        if len(real_points) == 3:
            interior_triangles.append(tri)

    return interior_triangles


if __name__ == "__main__":
#    n = 100
#    points = sgeom.MultiPoint([np.random.random(2) for _ in range(n)])

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

    c = alphabet.letter_to_polygon('e')

#    dt = DelaunayTriangulation(lake_michigan)
#    dt = DelaunayTriangulation(lake_superior)
    dt = ConstrainedDelaunayTriangulation(star)
#    dt = ConstrainedDelaunayTriangulation(c)

    fig = plt.figure()
    fig.add_subplot(1, 2, 1)
    for t in dt.triangles:
        plt.plot(*t.exterior.xy, linewidth=2, color='black')
    #for e in dt.edges:
    #    plt.plot(*e.xy, linewidth=2, color='blue')
    for p in dt.points:
        plt.plot(*p.buffer(0.01).exterior.xy, color='red', alpha=0.5)
    fig.add_subplot(1, 2, 2)
    triangles = shapely_triangulate(dt.points)
    for t in triangles:
        plt.plot(*t.exterior.xy, linewidth=2, color='black')
    for p in dt.points:
        plt.plot(*p.buffer(0.01).exterior.xy, color='red', alpha=0.5)
    plt.show()
