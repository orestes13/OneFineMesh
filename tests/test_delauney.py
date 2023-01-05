import string

import shapely.geometry as sgeom
from shapely.ops import unary_union
from shapely import affinity

from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt


def letter_to_polygon(letter, **kwargs):
    '''
    Construct a Polygon from a letter.

    Arguments:
        letter: the letter to be converted to a polygon

    Keyword arguments:
        anchor:
        scale:
        font_properties: font properties
    '''
    # parse kwargs
    anchor = kwargs.get('anchor', (0, 0))
    scale = kwargs.get('scale', (1, 1))
    fp = kwargs.get('font_properties', FontProperties(family='DejaVu Sans',
                                                      weight='bold'))

    # create TextPath, and convert to Polygon
    let = TextPath(anchor, letter, size=1, prop=fp)
    polygons = [sgeom.Polygon(pol) for pol in let.to_polygons()]
    geometry = sgeom.Polygon(polygons[0].boundary.coords,
                             holes=[pol.boundary.coords for pol in polygons[1:]])

    geometry = affinity.scale(geometry, xfact=scale[0], yfact=scale[1])

    return geometry

@pytest.mark.parametrize("letter", ["a", "b", "c"])
#@pytest.mark.parametrize("letter", list(string.ascii_lowercase))
def test_delauney_alphabet(test_input, expected):
    letter_to_polygon(letter, scale=(2, 1))
    assert True


if __name__ == "__main__":
    pass
#    a = letter_to_polygon(letter, scale=(2, 1))

#    plt.plot(*a.exterior.xy, marker='o')
#    for interior in a.interiors:
#        plt.plot(*interior.xy, marker='o')
#    plt.show()
