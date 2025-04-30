Okay, here's a Python Point3D class optimized for storage using __slots__. This technique prevents the creation of an instance __dict__, significantly reducing memory usage per object, especially when you have many instances.
import math # For potential future methods like distance calculation

class Point3D:
    """
    Represents a point in 3D space (x, y, z).

    Optimized for storage using __slots__.
    Coordinates are stored as floats.
    """
    # __slots__ prevents the creation of __dict__ for each instance, saving memory.
    # It pre-declares the instance attributes.
    __slots__ = ('x', 'y', 'z')

    def __init__(self, x: float, y: float, z: float):
        """
        Initializes a Point3D object.

        Args:
            x: The x-coordinate (float).
            y: The y-coordinate (float).
            z: The z-coordinate (float).

        Raises:
            TypeError: If any coordinate is not a number (int or float).
        """
        # Ensure coordinates are numeric, then explicitly convert to float
        if not isinstance(x, (int, float)):
            raise TypeError(f"x coordinate must be numeric, not {type(x).__name__}")
        if not isinstance(y, (int, float)):
            raise TypeError(f"y coordinate must be numeric, not {type(y).__name__}")
        if not isinstance(z, (int, float)):
            raise TypeError(f"z coordinate must be numeric, not {type(z).__name__}")

        self.x: float = float(x)
        self.y: float = float(y)
        self.z: float = float(z)

    def __repr__(self) -> str:
        """Returns an unambiguous string representation of the point."""
        # Use standard attribute access; it works fine with slots
        return f"Point3D(x={self.x}, y={self.y}, z={self.z})"

    def __str__(self) -> str:
        """Returns a user-friendly string representation of the point."""
        return f"({self.x}, {self.y}, {self.z})"

    def __eq__(self, other) -> bool:
        """Checks if two Point3D objects represent the same point."""
        if not isinstance(other, Point3D):
            return NotImplemented # Indicate comparison isn't supported for other types
        return self.x == other.x and self.y == other.y and self.z == other.z

    def __hash__(self) -> int:
        """
        Computes a hash value for the point.

        Allows Point3D objects to be used in sets or as dictionary keys.
        Be aware of potential precision issues when hashing floats.
        """
        return hash((self.x, self.y, self.z))

    # --- Optional useful methods ---

    # def distance_to(self, other: 'Point3D') -> float:
    #     """Calculates the Euclidean distance to another Point3D."""
    #     if not isinstance(other, Point3D):
    #         raise TypeError("Can only calculate distance to another Point3D object")
    #     dx = self.x - other.x
    #     dy = self.y - other.y
    #     dz = self.z - other.z
    #     return math.sqrt(dx*dx + dy*dy + dz*dz)

# --- Example Usage ---
if __name__ == "__main__":
    p1 = Point3D(1.0, 2.5, -3.0)
    p2 = Point3D(1, 2.5, -3)  # Integers are converted to floats
    p3 = Point3D(0, 0, 0)

    print(f"Point 1: {p1}")        # Uses __str__
    print(f"Point 1 repr: {repr(p1)}") # Uses __repr__
    print(f"Point 2: {p2}")
    print(f"Point 3: {p3}")

    print(f"Are p1 and p2 equal? {p1 == p2}") # True
    print(f"Are p1 and p3 equal? {p1 == p3}") # False

    # Using points in a set (requires __eq__ and __hash__)
    point_set = {p1, p2, p3}
    print(f"Set of points: {point_set}") # p1 and p2 are considered the same

    # Attempting to add an attribute not in __slots__ will fail
    try:
        p1.label = "Origin"
    except AttributeError as e:
        print(f"\nError as expected: {e}")

    # Memory usage check (conceptual - actual values depend on Python version/OS)
    # import sys
    # p_standard = type('Point3D_standard', (), {'__init__': lambda s,x,y,z: setattr(s,'x',x) or setattr(s,'y',y) or setattr(s,'z',z)})(1.0, 2.0, 3.0)
    # p_slotted = Point3D(1.0, 2.0, 3.0)
    # print(f"\nApprox memory (standard): {sys.getsizeof(p_standard) + sys.getsizeof(p_standard.__dict__)}")
    # print(f"Approx memory (slotted):  {sys.getsizeof(p_slotted)}") # Noticeably smaller

Explanation:
 * __slots__ = ('x', 'y', 'z'): This is the key optimization. It tells Python not to create a __dict__ for each instance. Instead, it allocates just enough space to hold references to the x, y, and z attributes directly in the object's structure.
 * __init__:
   * Takes x, y, z as arguments.
   * Includes basic isinstance checks to ensure inputs are numeric (integers or floats).
   * Crucially, it converts the inputs to float using float() before assigning them to self.x, self.y, self.z. This guarantees they are stored as floats as requested.
 * __repr__: Provides a developer-friendly representation that ideally allows recreating the object (e.g., Point3D(x=1.0, y=2.5, z=-3.0)).
 * __str__: Provides a more user-friendly, compact string representation (e.g., (1.0, 2.5, -3.0)).
 * __eq__: Defines how to check if two Point3D objects are equal (based on their coordinates). Necessary if you want to compare points using ==.
 * __hash__: Allows Point3D objects to be used in hash-based collections like sets or as keys in dictionaries. It's important to define __hash__ if you define __eq__.
 * Type Hinting (: float, -> str): Improves code readability and allows static analysis tools to check types.
 * No Arbitrary Attributes: Because __slots__ is used, you cannot add new attributes to instances of Point3D after they are created (e.g., p1.color = 'red' would raise an AttributeError). This is a trade-off for the memory savings.
This class effectively represents 3D points using floats while minimizing the memory overhead per instance.
