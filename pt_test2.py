import open3d as o3d
from open3d.web_visualizer import draw
cube_red = o3d.geometry.TriangleMesh.create_box(1, 2, 4)
cube_red.compute_vertex_normals()
cube_red.paint_uniform_color((1.0, 0.0, 0.0))
draw(cube_red)
cube_blue = o3d.geometry.TriangleMesh.create_box(1, 2, 4)
cube_blue.compute_vertex_normals()
cube_blue.paint_uniform_color((0.0, 0.0, 1.0))
draw(cube_blue)