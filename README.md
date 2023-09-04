![depth from obj usage graph](https://raw.githubusercontent.com/dwringer/depth-from-obj-node/main/depth_from_obj_usage.jpg)

### Depth Map from Wavefront OBJ

Render depth maps from Wavefront .obj files (triangulated) using this simple 3D renderer utilizing numpy and matplotlib to compute and color the scene. 

To be imported, an .obj must use triangulated meshes, so make sure to enable that option if exporting from a 3D modeling program. This renderer makes each triangle a solid color based on its average depth, so it will cause anomalies if your .obj has large triangles. In Blender, the Remesh modifier can be helpful to subdivide a mesh into small pieces that work well given these limitations.

There are simple parameters to change the FOV, camera position, and model orientation.

Additional parameters like different image sizes will probably be added, and things like more sophisticated rotations are planned but this node is experimental and may or may not change much.