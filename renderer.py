# John Tweedy
# 10/16/2023
# Modelling with Transformations

from PIL import Image, ImageDraw, ImageColor
import math
import numpy as np

# Classes
class SceneObject:
    def __init__(self):
        self.parent = None
        self.transform = np.eye(4)  # Identity matrix by default
        self.children = []  # New attribute to store child objects

    def global_transform(self):
        if self.parent:
            return np.dot(self.parent.global_transform(), self.transform)
        else:
            return self.transform

class Sphere(SceneObject):
    def __init__(self, center, radius, color, name):
        super().__init__()
        self.center = center
        self.radius = radius
        self.color = color
        self.name = name

class Plane(SceneObject):
    def __init__(self, point, normal, color, name):
        super().__init__()
        self.point = point
        self.normal = normal
        self.color = color
        self.name = name

class Triangle(SceneObject):
    def __init__(self, v0, v1, v2, color, name):
        super().__init__()
        self.v0 = v0
        self.v1 = v1
        self.v2 = v2
        self.color = color
        self.name = name

# Functions
# Ray-sphere intersection test
def ray_sphere_intersection(ray_origin, ray_direction, sphere):
    # Transform the sphere's center using its global transformation matrix
    transformed_center = np.dot(sphere.global_transform(), np.append(sphere.center, 1))[:3]
    
    oc = ray_origin - transformed_center
    a = np.dot(ray_direction, ray_direction)
    b = 2 * np.dot(oc, ray_direction)
    c = np.dot(oc, oc) - sphere.radius * sphere.radius
    discriminant = b * b - 4 * a * c
    
    if discriminant > 0:
        t1 = (-b - math.sqrt(discriminant)) / (2 * a)
        t2 = (-b + math.sqrt(discriminant)) / (2 * a)
        return min(t1, t2)
    return None


# Ray-plane intersection test
def ray_plane_intersection(ray_origin, ray_direction, plane):
    # Transform the ray_origin and ray_direction using the inverse of the plane's global transformation matrix
    inv_transform = np.linalg.inv(plane.global_transform())
    transformed_ray_origin = np.dot(inv_transform, np.append(ray_origin, 1))[:3]
    transformed_ray_direction = np.dot(inv_transform, np.append(ray_direction, 0))[:3]
    
    d = -np.dot(plane.normal, plane.point)
    denom = np.dot(plane.normal, transformed_ray_direction)
    
    if denom != 0:
        t = -(np.dot(plane.normal, transformed_ray_origin) + d) / denom
        if t >= 0:
            return t
    return None


# Ray-triangle intersection test
def ray_triangle_intersection(ray_origin, ray_direction, triangle):
    # Transform the triangle's vertices using its global transformation matrix
    transformed_v0 = np.dot(triangle.global_transform(), np.append(triangle.v0, 1))[:3]
    transformed_v1 = np.dot(triangle.global_transform(), np.append(triangle.v1, 1))[:3]
    transformed_v2 = np.dot(triangle.global_transform(), np.append(triangle.v2, 1))[:3]
    
    e1 = transformed_v1 - transformed_v0
    e2 = transformed_v2 - transformed_v0
    h = np.cross(ray_direction, e2)
    a = np.dot(e1, h)
    
    if a > -1e-6 and a < 1e-6:
        return None
    
    f = 1.0 / a
    s = ray_origin - transformed_v0
    u = f * np.dot(s, h)
    
    if u < 0 or u > 1:
        return None
    
    q = np.cross(s, e1)
    v = f * np.dot(ray_direction, q)
    
    if v < 0 or u + v > 1:
        return None
    
    t = f * np.dot(e2, q)
    if t > 1e-6:
        return t
    return None


def translation_matrix(tx, ty, tz):
    return np.array([
        [1, 0, 0, tx],
        [0, 1, 0, ty],
        [0, 0, 1, tz],
        [0, 0, 0, 1]
    ])

def rotation_matrix(rx, ry, rz):
    cos_x = math.cos(math.radians(rx))
    sin_x = math.sin(math.radians(rx))
    cos_y = math.cos(math.radians(ry))
    sin_y = math.sin(math.radians(ry))
    cos_z = math.cos(math.radians(rz))
    sin_z = math.sin(math.radians(rz))

    rot_x = np.array([
        [1, 0, 0, 0],
        [0, cos_x, -sin_x, 0],
        [0, sin_x, cos_x, 0],
        [0, 0, 0, 1]
    ])

    rot_y = np.array([
        [cos_y, 0, sin_y, 0],
        [0, 1, 0, 0],
        [-sin_y, 0, cos_y, 0],
        [0, 0, 0, 1]
    ])

    rot_z = np.array([
        [cos_z, -sin_z, 0, 0],
        [sin_z, cos_z, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    return rot_x @ rot_y @ rot_z

# Transform Object
def transform_object(obj_name, translation=(0, 0, 0), rotation=(0, 0, 0)):
    obj = next((o for o in objects if o.name == obj_name), None)
    if not obj:
        print(f"Object with name {obj_name} not found.")
        return

    tx, ty, tz = translation
    rx, ry, rz = rotation
    T = translation_matrix(tx, ty, tz)
    R = rotation_matrix(rx, ry, rz)

    # Assuming pivot is the center of the car.
    pivot = obj.center
    T_pivot_to_origin = translation_matrix(-pivot[0], -pivot[1], -pivot[2])
    T_origin_to_pivot = translation_matrix(pivot[0], pivot[1], pivot[2])

    # Translate to origin, Rotate, then Translate back
    obj.transform = obj.transform @ T_origin_to_pivot @ R @ T_pivot_to_origin @ T



def render():
    global closest_t
    global closest_color

    # Initialize variables for the closest object hit
    closest_t = float("inf")
    closest_color = None

    # Create an image to render
    img = Image.new("RGB", (image_width, image_height), color="cyan")
    draw = ImageDraw.Draw(img)

    # Define the ray origin (camera position)
    ray_origin = eye

    # Trace rays and render the scene
    for y in range(image_height):
        for x in range(image_width):
            # Calculate the ray direction based on the camera parameters
            aspect_ratio = image_width / image_height
            x_normalized = (2 * (x + 0.5) / image_width - 1) * math.tan(math.radians(fov / 2)) * aspect_ratio
            y_normalized = (1 - 2 * (y + 0.5) / image_height) * math.tan(math.radians(fov / 2))
            ray_direction = np.array([x_normalized, y_normalized, -1])
            ray_direction /= np.linalg.norm(ray_direction)

            # Reset closest object hit variables
            closest_t = float("inf")
            closest_color = None

            # Ray-object intersection tests
            for obj in objects:
                if isinstance(obj, Sphere):
                    t = ray_sphere_intersection(ray_origin, ray_direction, obj)
                elif isinstance(obj, Plane):
                    t = ray_plane_intersection(ray_origin, ray_direction, obj)
                elif isinstance(obj, Triangle):
                    t = ray_triangle_intersection(ray_origin, ray_direction, obj)
                else:
                    t = None

                if t is not None and t < closest_t:
                    closest_t = t
                    closest_color = obj.color

            # Color the pixel based on the closest intersection
            if closest_color:
                pixel_color = ImageColor.getrgb(closest_color)
                draw.point((x, y), fill=pixel_color)

    # Save the rendered image
    img.save("renderedScene.png")

# Initialize default values if missing from scene file
image_width, image_height = 800, 600
viewport = None
eye = None
objects = []
fov = 60

# Read and parse scene information from scene.txt
with open("scene.txt", "r") as scene_file:
    current_object = None  # to keep track of the current object being parsed

    # Read each line except for comment lines
    for line in scene_file:
        line = line.strip()
        if line.startswith("#") or not line:
            continue
        parts = line.split()
        if not parts:
            continue
        tag = parts[0]

        if tag == "image":
            image_width, image_height = int(parts[1]), int(parts[2])
        elif tag == "viewport":
            viewport = list(map(float, parts[1:]))
        elif tag == "eye":
            eye = np.array(list(map(float, parts[1:])))
        elif tag == "sphere":
            center = np.array(list(map(float, parts[1:4])))
            radius = float(parts[4])
            color = parts[5]
            name = parts[6]
            current_object = Sphere(center, radius, color, name)
            objects.append(current_object)
        elif tag == "plane":
            point = np.array(list(map(float, parts[1:4])))
            normal = np.array(list(map(float, parts[4:7])))
            color = parts[7]
            name = parts[8]
            current_object = Plane(point, normal, color, name)
            objects.append(current_object)
        elif tag == "tri":
            v0 = np.array(list(map(float, parts[1:4])))
            v1 = np.array(list(map(float, parts[4:7])))
            v2 = np.array(list(map(float, parts[7:10])))
            color = parts[10]
            name = parts[11]
            current_object = Triangle(v0, v1, v2, color, name)
            objects.append(current_object)
        elif tag == "transform":
            current_object.transform = np.array(list(map(float, parts[1:]))).reshape(4, 4)
        elif tag == "parent":
            parent_name = parts[1]
            if parent_name != "null":
                parent_object = next((obj for obj in objects if obj.name == parent_name), None)
                current_object.parent = parent_object
                # Link the child to the parent
                parent_object.children.append(current_object)

# Print objects for confirmation
# for obj in objects:
#    print(obj.__class__.__name__, vars(obj))

# Automatically call render after setup is complete
render()

# Main loop for user input
while True:
    # Get user input
    command = input("Enter command (W, S, A, D, Q to quit): ").upper()

    # Check the command and perform corresponding action
    if command == 'W':
        # Forward movement in car's reference frame
        transform_object("CarBody", translation=(0, 0, -5))
    elif command == 'S':
        # Backward movement in car's reference frame
        transform_object("CarBody", translation=(0, 0, 5))
    elif command == 'A':
        # Rotate left in car's reference frame
        transform_object("CarBody", rotation=(0, 45, 0))  # -5 degrees
    elif command == 'D':
        # Rotate right in car's reference frame
        transform_object("CarBody", rotation=(0, -45, 0))  # 5 degrees
    elif command == 'Q':
        # Quit the loop
        break
    else:
        print("Invalid command. Please enter W, S, A, D or Q.")

    # Render and save the scene after each command
    render()
