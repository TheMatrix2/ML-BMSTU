import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Path, PathPatch, Polygon
from matplotlib.path import Path as MPath
import random


def draw_rounded_polygon(ax, num_sides, side_length, center=(0, 0), rotation_angle=0, color='cyan', corner_radius=0.1):
    rotation_rad = np.radians(rotation_angle)

    R = side_length / (2 * np.sin(np.pi / num_sides))

    angles = np.linspace(0, 2 * np.pi, num_sides, endpoint=False) + rotation_rad
    vertices = []
    for angle in angles:
        x = center[0] + R * np.cos(angle)
        y = center[1] + R * np.sin(angle)
        vertices.append((x, y))

    path_vertices = []
    path_codes = []

    safe_radius = min(side_length / 2, side_length * np.sin(np.pi / num_sides))
    actual_radius = corner_radius * safe_radius

    for i in range(num_sides):
        current = np.array(vertices[i])
        prev = np.array(vertices[i - 1])
        next_vertex = np.array(vertices[(i + 1) % num_sides])

        to_prev = prev - current
        to_next = next_vertex - current

        to_prev_norm = to_prev / np.linalg.norm(to_prev) * actual_radius
        to_next_norm = to_next / np.linalg.norm(to_next) * actual_radius

        start_point = current + to_prev_norm
        end_point = current + to_next_norm

        if i == 0:
            path_vertices.append(start_point)
            path_codes.append(MPath.MOVETO)
        else:
            path_vertices.append(start_point)
            path_codes.append(MPath.LINETO)

        path_vertices.append(current)
        path_codes.append(MPath.CURVE3)

        path_vertices.append(end_point)
        path_codes.append(MPath.CURVE3)

    path_vertices.append(path_vertices[0])
    path_codes.append(MPath.LINETO)

    path = Path(path_vertices, path_codes)
    patch = PathPatch(path, facecolor=color, alpha=0.9, edgecolor=color, lw=1)
    ax.add_patch(patch)

    return R


def draw_star(ax, center, size, color, rotation_angle=0):
    rotation_rad = np.radians(rotation_angle)

    n_points = 5

    inner_radius = size * 0.4
    outer_radius = size

    all_angles = np.linspace(0, 2 * np.pi, 2 * n_points, endpoint=False) + rotation_rad
    radii = np.array([outer_radius, inner_radius] * n_points)

    points = []
    for angle, radius in zip(all_angles, radii):
        x = center[0] + radius * np.cos(angle)
        y = center[1] + radius * np.sin(angle)
        points.append((x, y))

    polygon = Polygon(points, closed=True, facecolor=color, edgecolor=color, lw=1, alpha=0.9)
    ax.add_patch(polygon)

    return outer_radius


def check_overlap(objects, new_center, new_radius, min_distance=1.0):
    for obj in objects:
        center_x, center_y, radius = obj
        distance = np.sqrt((center_x - new_center[0]) ** 2 + (center_y - new_center[1]) ** 2)
        if distance < (radius + new_radius) * min_distance:
            return True
    return False


def random_red_hex():
    return f'#{random.randint(200, 255):02X}{random.randint(0, 50):02X}{random.randint(0, 50):02X}'


def random_yellow_hex():
    return f'#{random.randint(200, 255):02X}{random.randint(200, 255):02X}{random.randint(0, 50):02X}'


def random_blue_hex():
    return f'#{random.randint(0, 50):02X}{random.randint(0, 50):02X}{random.randint(200, 255):02X}'


def create_shapes_image(save_path=None, show=True, img_width=1920, img_height=1080, min_distance=1.2):
    img_width, img_height = img_width / 100, img_height / 100
    shape_types = ['star', 'triangle', 'square', 'pentagon']

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(img_width, img_height))
    fig.patch.set_facecolor('black')
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set_facecolor('black')

    ax.set_xlim(-img_width / 2, img_width / 2)
    ax.set_ylim(-img_height / 2, img_height / 2)
    ax.set_aspect('equal')

    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    objects = []
    placed_objects = []

    while len(placed_objects) < random.randint(20, 40):
        shape_type = shape_types[len(placed_objects) % len(shape_types)]
        colors = {'red': random_red_hex(), 'yellow': random_yellow_hex(), 'blue': random_blue_hex()}
        for color_name, color_hex in colors.items():
            lim_attempts = 100
            placed = False

            for _ in range(lim_attempts):
                size = random.uniform(0.5, 1.5)
                rotation = random.uniform(0, 360)

                margin = size
                x = random.uniform(-img_width / 2 + margin, img_width / 2 - margin)
                y = random.uniform(-img_height / 2 + margin, img_height / 2 - margin)

                if not check_overlap(objects, (x, y), size, min_distance):
                    if shape_type == 'star':
                        radius = draw_star(ax, (x, y), size, color_hex, rotation)
                    elif shape_type == 'triangle':
                        radius = draw_rounded_polygon(ax, 3, size, (x, y), rotation, color_hex, 0.2)
                    elif shape_type == 'square':
                        radius = draw_rounded_polygon(ax, 4, size, (x, y), rotation, color_hex, 0.15)
                    elif shape_type == 'pentagon':
                        radius = draw_rounded_polygon(ax, 5, size, (x, y), rotation, color_hex, 0.12)

                    objects.append((x, y, radius))
                    placed_objects.append((shape_type, color_name, x, y, size, rotation))
                    placed = True
                    break

            if not placed:
                print(f"Не удалось разместить {color_name} {shape_type} после {lim_attempts} попыток")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='black')

    if show:
        plt.show()

    return fig, ax, placed_objects


if __name__ == '__main__':
    fig, ax, objects = create_shapes_image(
        save_path='img.png',
        show=True,
        min_distance=1.6
    )

    print(f"Создано объектов: {len(objects)}")

    for i, obj in enumerate(objects):
        shape, color, x, y, size, rotation = obj
        print(f"{i + 1}. {color} {shape}: координаты ({x:.2f}, {y:.2f}), размер {size:.2f}, поворот {rotation:.2f}°")
