'''
Constrained Delaunay Triangulations

David Bentley
27/03/2018
'''

import numpy as np
import shapely.geometry as sgeom
from shapely.ops import unary_union
from shapely.ops import triangulate as shapely_triangulate
from shapely.affinity import scale
from itertools import combinations
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
        self.points = shape


class ConstrainedDelaunayTriangulation(object):
    def __init__(self, shape):
        '''
        Construct a constrained Delaunay triangulation of the provided
        shape

        Arguments:
            shape: a shapely.geometry defining the shape to be triangulated
        '''
        return None


class ConformalDelaunayTriangulation(object):
    def __init__(self, shape):
        '''
        Construct a conformal Delaunay triangulation of the provided
        shape

        Arguments:
            shape: a shapely.geometry defining the shape to be triangulated
        '''
        return None

# 3D tetrahedralisations
#class DelaunayTetrahedralisation(object)

# generic functions
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
    vertices = zip(*triangle.exterior.xy)
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


def triangular_bounding_box(points, scale_factor=3.0):
    '''
    Given the convex hull of a set of points, return a triangle containing
    all the points.

    Arguments:
        points: a shapely.geometry.MultiPoint of the popitns to be bounded

    Keyword arguments:
        scale_factor: the scale factor by which to increase the radius of
                      the enclosing circle (>= 1)

    Note:
        to construct the triangle, we first find the distance from the centre
        of mass to the furthest point, and call it r. We then imagine a
        circle of radius (1 + epsilon) * r, centred on the centre of mass.
        The triangle is then the equilateral triangle that is tangential
        to this imagined circle at three points, spaced 120 degrees apart.
        The length of the triangle edges, 2 * np.sqrt(3) * r * (1 + eps),
        follows from simple geometrical arguments.
    '''
    # check that scale_factor is >= 1
    if not scale_factor >= 1:
        raise ValueError('scale_factor {} is < 1'.format(scale_factor))

    # determine the centre of mass of the points
    centre = points.convex_hull.centroid

    # determine distance to furthest point
    radius = scale_factor * max([np.linalg.norm(np.array(centre) - np.array(p)) for p in points])

    # construct a triangle enclosing the circle
    midpoint = sgeom.Point(np.array(centre) - np.array([0, radius]))
    bound = sgeom.Polygon([np.array(midpoint) - np.array([np.sqrt(3) * radius, 0]),
                           np.array(midpoint) + np.array([np.sqrt(3) * radius, 0]),
                           np.array(centre) + np.array([0, 2 * radius])])
    return bound


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


def new_triangulate(points):
    '''
    Construct the Delaunay triangulation of a set of points

    Arguments:
        points: a shapely.geometry.MultiPoint defining the points to be
                triangulated
    '''
    triangles = []

    # choose three random points in the point set to form an initial triangle
    inserted_points = []
    for _ in range(3):
        p = random.choice(points)
        points = points.difference(p)
        inserted_points.append(p)
    initial_triangle = sgeom.Polygon([(p.x, p.y) for p in inserted_points])
    triangles.append(initial_triangle)

    # calculate the ghost triangles
    triangles.extend(ghost_triangles(sgeom.MultiPoint(inserted_points)))

    # loop through the remaining points in a random order, inserting them
    # into the bounding triangle and constructing the triangulation
    while len(points) > 0:
        p = random.choice(points)
#    for p in points:
        print(len(points))
        try:
            points = sgeom.MultiPoint(points.difference(p))
        except TypeError:
            points = sgeom.MultiPoint([points.difference(p)])

        # determine if the new point is inside the point set
        convex_hull = sgeom.MultiPoint(inserted_points).convex_hull
        interior_point = p.within(convex_hull)

        triangles_to_remove = np.zeros(len(triangles), dtype=bool)
        for idx, tri in enumerate(triangles):
            ghost_vertices = []
            tri_points = list(set(list(tri.exterior.coords)))
            real_points = [sgeom.Point(tp) in inserted_points for tp in tri_points]
            if all(real_points):
                # normal triangle
                c = circumcircle(tri, None)
                if p.within(c):
                    triangles_to_remove[idx] = True
            else:
                # ghost triangle
                ghost_vertex = tri_points[np.squeeze(np.where(~np.array(real_points)))]
                ghost_normal, edge = ghost_circumcircle(tri, sgeom.Point(ghost_vertex))
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

        # construct the union of the containing triangles, and remove
        # from the triangulation
        good_triangles = list(np.array(triangles)[~triangles_to_remove])
        bad_triangles = list(np.array(triangles)[triangles_to_remove])
        containing_union = unary_union(bad_triangles)

#        fig = plt.figure(figsize=(18, 10))
#        fig.add_subplot(1, 2, 1)
#        plt.plot(*p.buffer(0.02).exterior.xy, color='blue', alpha=0.8)
#        for ttt in good_triangles:
#            plt.plot(*ttt.exterior.xy, linewidth=1, color='black', alpha=0.7)
#        for ttt in bad_triangles:
#            plt.plot(*ttt.exterior.xy, linewidth=1, color='blue', alpha=0.7)
#        for ppp in inserted_points:
#            plt.plot(*ppp.buffer(0.01).exterior.xy, color='red', alpha=0.8)
#        plt.xlim([-1, 2])
#        plt.ylim([-1, 2])
#        fig.add_subplot(1, 2, 2)
#        if isinstance(containing_union, sgeom.Polygon):
#            plt.plot(*containing_union.exterior.xy, linewidth=2, color='green')
#        elif isinstance(containing_union, sgeom.MultiPolygon):
#            for pol in containing_union:
#                plt.plot(*pol.exterior.xy, linewidth=2, color='green')
#        for ppp in inserted_points:
#            plt.plot(*ppp.buffer(0.01).exterior.xy, color='red', alpha=0.8)
#        plt.plot(*p.buffer(0.02).exterior.xy, color='blue', alpha=0.8)
#        plt.xlim([-1, 2])
#        plt.ylim([-1, 2])
#        plt.savefig('good_bad_triangles_{}.png'.format(len(points)), dpi=100, bbox_inches='tight')
#        plt.close()

#        old_triangles = list(np.array(triangles)[~triangles_to_remove])

        # remove the containing shape, and construct new triangles
        inserted_points.append(p)
        if isinstance(containing_union, sgeom.Polygon):
            if isinstance(containing_union.boundary, sgeom.MultiLineString):
                for line in containing_union.boundary:
                    print line
            for i, x in enumerate(containing_union.boundary.coords[:-1]):
                new_tri = sgeom.Polygon([x, containing_union.boundary.coords[i +  1], np.array(p)])
                tri_points = list(set(list(new_tri.exterior.coords)))
                real_points = [sgeom.Point(tp) in inserted_points for tp in tri_points]
                if all(real_points):
#                    area_of_intersection = sum([new_tri.intersection(ttt).area for ttt in old_triangles])
#                    if area_of_intersection > 0:
#                        plt.plot(*new_tri.exterior.xy, linewidth=1, color='black')
#                        for ppp in inserted_points:
#                            plt.plot(*ppp.buffer(0.005).exterior.xy, color='green')
#                        plt.plot(*p.buffer(0.005).exterior.xy, color='blue')
#                        #for ttt in good_triangles:
#                        #    xx, yy = ttt.exterior.coords.xy
#                        #    pxy = np.array([xx, yy]).T
#                        #    polygon_shape = patches.Polygon(pxy, facecolor='red', edgecolor='none', alpha=0.3)
#                        #    plt.gca().add_patch(polygon_shape)
#                        for ttt in bad_triangles:
#                            xx, yy = ttt.exterior.coords.xy
#                            pxy = np.array([xx, yy]).T
#                            polygon_shape = patches.Polygon(pxy, edgecolor='none', alpha=0.3)
#                            plt.gca().add_patch(polygon_shape)
#                        for ttt in old_triangles:
#                            if new_tri.intersection(ttt).area > 0:
#                                plt.plot(*ttt.exterior.xy, linewidth=1, color='black', alpha=0.5)
#                                ccc = circumcircle(ttt, 64)
#                                plt.plot(*ccc.exterior.xy, linewidth=1, color='black', alpha=0.5)
#                            else:
#                                plt.plot(*ttt.exterior.xy, linewidth=1, color='black', alpha=0.1)
#                        plt.xlim([-1, 2])
#                        plt.ylim([-1, 2])
#                        plt.show()
#                    #print('area of intersection', area_of_intersection)
                    good_triangles.append(new_tri)
        elif isinstance(containing_union, sgeom.MultiPolygon):
            for pol in containing_union:
                if isinstance(pol.boundary, sgeom.MultiLineString):
                    for line in pol.boundary:
                        print line
                for i, x in enumerate(pol.boundary.coords[:-1]):
                    new_tri = sgeom.Polygon([x, pol.boundary.coords[i +  1], np.array(p)])
                    tri_points = list(set(list(new_tri.exterior.coords)))
                    real_points = [sgeom.Point(tp) in inserted_points for tp in tri_points]
                    if all(real_points):
#                        area_of_intersection = sum([new_tri.intersection(ttt).area for ttt in old_triangles])
#                        if area_of_intersection > 0:
#                            plt.plot(*new_tri.exterior.xy, linewidth=1, color='black')
#                            for ppp in inserted_points:
#                                plt.plot(*ppp.buffer(0.005).exterior.xy, color='green')
#                            plt.plot(*p.buffer(0.005).exterior.xy, color='blue')
#                            #for ttt in good_triangles:
#                            #    xx, yy = ttt.exterior.coords.xy
#                            #    pxy = np.array([xx, yy]).T
#                            #    polygon_shape = patches.Polygon(pxy, facecolor='red', edgecolor='none', alpha=0.3)
#                            #    plt.gca().add_patch(polygon_shape)
#                            for ttt in bad_triangles:
#                                xx, yy = ttt.exterior.coords.xy
#                                pxy = np.array([xx, yy]).T
#                                polygon_shape = patches.Polygon(pxy, edgecolor='none', alpha=0.3)
#                                plt.gca().add_patch(polygon_shape)
#                            for ttt in old_triangles:
#                                if new_tri.intersection(ttt).area > 0:
#                                    plt.plot(*ttt.exterior.xy, linewidth=1, color='black', alpha=0.5)
#                                    ccc = circumcircle(ttt, 64)
#                                    plt.plot(*ccc.exterior.xy, linewidth=1, color='black', alpha=0.5)
#                                else:
#                                    plt.plot(*ttt.exterior.xy, linewidth=1, color='black', alpha=0.1)
#                            plt.xlim([-1, 2])
#                            plt.ylim([-1, 2])
#                            plt.show()
#                        #print('area of intersection', area_of_intersection)
                        good_triangles.append(new_tri)

        # recompute ghost triangles
        ghosts = ghost_triangles(sgeom.MultiPoint(inserted_points))
        for tri in ghosts:
            if tri not in good_triangles:
                good_triangles.append(tri)

        triangles = good_triangles

    # reset points
    points = inserted_points

    # remove ghost triangles
    interior_triangles = []
    for tri in triangles:
        tri_points = list(set(list(tri.exterior.coords)))
        real_points = [sgeom.Point(tp) in inserted_points for tp in tri_points]
        if all(real_points):
            interior_triangles.append(tri)

    return interior_triangles


def triangulate(points):
    '''
    Construct the Delaunay triangulation of a set of points

    Arguments:
        points: a shapely.geometry.MultiPoint defining the points to be
                triangulated
    '''
    # get the bounding triangle
    bounding_triangle = triangular_bounding_box(points)

    # loop through the points in a random order, inserting them into the
    # bounding triangle and constructing the triangulation
    triangles = [bounding_triangle]
    plot_points = []
    for p in points:
        plot_points.append(p)
        plt.plot(*p.buffer(0.1).exterior.xy, color='red')
        triangles_to_remove = np.zeros(len(triangles), dtype=bool)
        for idx, tri in enumerate(triangles):
            c = circumcircle(tri)
            if p.within(c):
                plt.plot(*tri.exterior.xy, linewidth=2, color='black')
                plt.plot(*c.exterior.xy, linewidth=2, color='black')
                triangles_to_remove[idx] = True
            else:
                plt.plot(*tri.exterior.xy, linewidth=2, color='gray')
                plt.plot(*c.exterior.xy, linewidth=0.7, color='gray')
        # construct the union of the containing triangles, and remove
        # from the triangulation
        good_triangles = list(np.array(triangles)[~triangles_to_remove])
        bad_triangles = list(np.array(triangles)[triangles_to_remove])
        containing_union = unary_union(bad_triangles)
#        print containing_union
        try:
            plt.plot(*containing_union.exterior.xy, linewidth=3, color='blue')
        except AttributeError:
            for poly in containing_union:
                plt.plot(*poly.exterior.xy, linewidth=3, color='blue')
        plt.show()
        # remove the containing shape, and construct new triangles
        for i, x in enumerate(containing_union.boundary.coords[:-1]):
            new_tri = sgeom.Polygon([x, containing_union.boundary.coords[i +  1],
                                     np.array(p)])
            good_triangles.append(new_tri)
        triangles = good_triangles
#        for pp in plot_points:
#            plt.plot(*pp.buffer(0.1).exterior.xy, color='red')
#        for tri in triangles:
#            plt.plot(*tri.exterior.xy, linewidth=2, color='black')
#        plt.show()
    # remove the triangles which have the vertices of the bounding
    # triangle as one of their vertices
    interior_triangles = []
    c_hull = points.convex_hull
    for tri in triangles:
        if c_hull.contains(tri):
            interior_triangles.append(tri)
    return interior_triangles


if __name__ == "__main__":
    for j in range(1):
        np.random.seed(j)
        n = 150
        points = sgeom.MultiPoint([np.random.random(2) for _ in range(n)])

        triangles = new_triangulate(points)
#    triangles = triangulate(points)

    fig = plt.figure()
    fig.add_subplot(1, 2, 1)
    for t in triangles:
        plt.plot(*t.exterior.xy, linewidth=2, color='black')
    for p in points:
        plt.plot(*p.buffer(0.01).exterior.xy, color='red', alpha=0.5)
    fig.add_subplot(1, 2, 2)
    triangles = shapely_triangulate(points)
    for t in triangles:
        plt.plot(*t.exterior.xy, linewidth=2, color='black')
    for p in points:
        plt.plot(*p.buffer(0.01).exterior.xy, color='red', alpha=0.5)
    plt.show()
