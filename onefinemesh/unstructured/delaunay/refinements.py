import numpy as np
import shapely.geometry as sgeom
import shapely.ops

import random

import matplotlib.pyplot as plt
from matplotlib import patches

def ruppert(triangulation, alpha=20):
    '''
    Refine the given Delaunay triangulation using Ruppert's algorithm.

    Arguments:
        triangulation: the triangulation to be refined

    Keyword arguments:
        alpha: the desired minimum angle bound (degrees)

    Note:
        if alpha > 20.7 then the algorithm is not guaranteed to terminate
    '''
    # get the PSLG segments and vertices from the triangulation
    segments = list(triangulation.segments)
    vertices = triangulation.points

    # determine the minimum angle in the triangulation
    min_angle = min([min(tri.angles) for tri in triangulation.triangles])
    skinny_triangles = [tri for tri in triangulation.triangles if any(np.array(tri.angles) < np.deg2rad(alpha))]

    # determine which of the segments are encroached
    encroached_segments = []
    for s in segments:
        cc, cr = diametral_circle(s)
        points_to_check = vertices.difference(s.boundary)
        if points_within_circle(np.array(vertices.difference(s.boundary)), cc, cr):
            encroached_segments.append(s)

    while len(encroached_segments) > 0 or len(skinny_triangles) > 0:
        # first split the encroached segments
        segments_to_delete = []
        while len(encroached_segments) > 0:
            print('encroached segments: {}'.format(len(encroached_segments)))
            s = encroached_segments.pop()
            # plot encroached segment
            #cc, cr = diametral_circle(s)
            #for pp in triangulation.points:
            #    if points_within_circle(np.array(pp).reshape((1, 2)), cc, cr):
            #        patch = patches.Polygon(np.array(pp.buffer(0.01).exterior.coords.xy).T,
            #                                facecolor='red', edgecolor='black')
            #        plt.gca().add_patch(patch)
            #    else:
            #        patch = patches.Polygon(np.array(pp.buffer(0.01).exterior.coords.xy).T,
            #                                facecolor='none', edgecolor='black')
            #        plt.gca().add_patch(patch)
            #for t in triangulation.triangles:
            #    plt.plot(*t.exterior.xy, color='gray', linestyle='--')
            #for seg in segments:
            #    plt.plot(*seg.xy, color='blue', alpha=0.3)
            #plt.plot(*s.xy, color='blue', linewidth=2)
            #plt.plot(*sgeom.Point(cc).buffer(cr, resolution=1024).exterior.coords.xy, linestyle='--', color='blue')
            #plt.xlim([0, 1])
            #plt.ylim([0, 1])
            #plt.show()

            segments_to_delete.append(s)
            midpoint = s.interpolate(0.5, normalized=True)
            triangulation.insert_point(midpoint)
            vertices = triangulation.points
            s1, s2 = split(s, midpoint)
            segments.extend([s1, s2])

            segments = [s for s in segments if s not in segments_to_delete]
            # update encroached segments
            encroached_segments = []
            for s in segments:
                cc, cr = diametral_circle(s)
                points_to_check = vertices.difference(s.boundary)
                if points_within_circle(np.array(vertices.difference(s.boundary)), cc, cr):
                    encroached_segments.append(s)

        print('no. of encroached subsegments: {}'.format(len(encroached_segments)))

        # find a triangle with an interior angle smaller than the given lower
        # bound, and split this triangle
        skinny_triangles = [tri for tri in triangulation.triangles if any(np.array(tri.angles) < np.deg2rad(alpha))]
        print('skinny_triangles', len(skinny_triangles))

        if len(skinny_triangles) > 0:
            triangle_to_split = skinny_triangles.pop()#random.choice(skinny_triangles)
            p = triangle_to_split.circumcentre
            encroaches = False
            for s in segments:
                cc, cr = diametral_circle(s)
                if points_within_circle(np.array(p).reshape((1, 2)), cc, cr):
                    encroached_segments.append(s)
                    encroaches = True
            if encroaches:
                skinny_triangles.append(triangle_to_split)
                continue
            else:
                triangulation.insert_point(p)
                vertices = triangulation.points
                skinny_triangles = [tri for tri in triangulation.triangles if any(np.array(tri.angles) < np.deg2rad(alpha))]
                min_angle = min([min(tri.angles) for tri in triangulation.triangles])

    return triangulation


def split(line, point):
    '''
    To workaround precision issues with shapely.ops.split

    Arguments:
        line: the segment to be split
        point: the point with thich to split the segment
    '''
    split_point = shapely.ops.nearest_points(line, point)[0]
    return (sgeom.LineString([line.boundary[0], split_point]),
            sgeom.LineString([split_point, line.boundary[1]]))

def points_within_circle(points, centre, radius, eps=1e-12):
    '''
    Determine whether the given point is within the circle defined
    by the given centre and radius.

    Arguments:
        point: the point(s) to check
        centre: the centre of the circle
        radius: the radius of the circle

    Keyword arguments:
        eps: tolerance for distance comparison
    '''
    distance = np.linalg.norm(points - centre, axis=1)
    return any(distance <= (1 - eps) * radius)


def diametral_circle(segment):
    '''
    Calculate the diametral circle of the given segment.

    Arguments:
        segment: a sgeom.LineString defining the segment for which to calculate
                 the diametral circle

    Note:
        the diametral circle is the smallest circle containing the segment (so
        centred on the midpoint of the segement, and whose diameter is the
        length of the segment)
    '''
    midpoint = np.array(segment.centroid)
    radius = 0.5 * segment.length
    return midpoint, radius

if __name__ =="__main__":
    pass
