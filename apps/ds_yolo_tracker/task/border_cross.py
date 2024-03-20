class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
  
# Given three collinear points p, q, r, the function checks if 
# point q lies on line segment 'pr' 
def onSegment(p, q, r):
    """
    Determines if point q lies on the line segment formed by points p and r.

    Args:
        p (Point): The starting point of the line segment.
        q (Point): The point to be checked.
        r (Point): The ending point of the line segment.

    Returns:
        bool: True if point q lies on the line segment, False otherwise.
    """
    if ( (q.x <= max(p.x, r.x)) and (q.x >= min(p.x, r.x)) and 
           (q.y <= max(p.y, r.y)) and (q.y >= min(p.y, r.y)) ):
        return True
    return False
  
def orientation(p, q, r):
    """
    Determines the orientation of an ordered triplet (p, q, r).

    Parameters:
    p (Point): The first point of the triplet.
    q (Point): The second point of the triplet.
    r (Point): The third point of the triplet.

    Returns:
    int: The orientation value.
        - 0: Collinear points
        - 1: Clockwise points
        - 2: Counterclockwise points

    Reference:
    See https://www.geeksforgeeks.org/orientation-3-ordered-points/ for details of the formula used.
    """
    val = (float(q.y - p.y) * (r.x - q.x)) - (float(q.x - p.x) * (r.y - q.y))
    if val > 0:
        return 1  # Clockwise orientation
    elif val < 0:
        return 2  # Counterclockwise orientation
    else:
        return 0  # Collinear orientation
  
# The main function that returns true if 
# the line segment 'p1q1' and 'p2q2' intersect.
def doIntersect(p1, q1, p2, q2):
    """
    Determines if two line segments intersect.

    Args:
        p1 (tuple): The coordinates of the first endpoint of the first line segment.
        q1 (tuple): The coordinates of the second endpoint of the first line segment.
        p2 (tuple): The coordinates of the first endpoint of the second line segment.
        q2 (tuple): The coordinates of the second endpoint of the second line segment.

    Returns:
        bool: True if the line segments intersect, False otherwise.
    """
    # Find the 4 orientations required for 
    # the general and special cases
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)
  
    # General case
    if ((o1 != o2) and (o3 != o4) and (o4 != 0)):
        print('general')
        return True
  
    # Special Cases
  
    # p1 , q1 and p2 are collinear and p2 lies on segment p1q1
    if ((o1 == 0) and onSegment(p1, p2, q1)):
        print('s1')
        return True
  
    # p1 , q1 and q2 are collinear and q2 lies on segment p1q1
    if ((o2 == 0) and onSegment(p1, q2, q1)):
        print('s2')
        return True
  
    # p2 , q2 and p1 are collinear and p1 lies on segment p2q2
    if ((o3 == 0) and onSegment(p2, p1, q2)):
        print('s3')
        return True
  
    # p2 , q2 and q1 are collinear and q1 lies on segment p2q2
    if ((o4 == 0) and onSegment(p2, q1, q2)):
        print('s4')
        return False 
  
    # If none of the cases
    return False

# check point in polygon
def isInside(polygon, n, p):
    """
    Determines whether a given point `p` is inside a polygon.

    Args:
        polygon (list): List of vertices of the polygon.
        n (int): Number of vertices in the polygon.
        p (Point): Point to be checked.

    Returns:
        bool: True if the point is inside the polygon, False otherwise.
    """
    # There must be at least 3 vertices in polygon[]
    if (n < 3):
        return False
  
    # Create a point for line segment from p to infinite
    extreme = Point(10000, p.y)
  
    # Count intersections of the above line with sides of polygon
    count = 0
    i = 0
    while(True):
        next = (i+1)%n
  
        # Check if the line segment from 'p' to 'extreme' intersects 
        # with the line segment from 'polygon[i]' to 'polygon[next]'
        if (doIntersect(polygon[i], polygon[next], p, extreme)):
              
            # If the point 'p' is colinear with line segment 'i-next', 
            # then check if it lies on segment. If it lies, return true, 
            # otherwise false 
            if (orientation(polygon[i], p, polygon[next]) == 0):
                return onSegment(polygon[i], p, polygon[next])
  
            count += 1
        i = next
        if (i == 0):
            break
        print(i, count)
  
    # Return true if count is odd, false otherwise 
    return count&1 # Same as (count%2 == 1)

polygon = [Point(1, 0), Point(8, 3), Point(8, 8), Point(1, 5)]

p = Point(3, 3)

if (isInside(polygon, len(polygon), p)):
    print("True")
else:
    print("False")
  
  