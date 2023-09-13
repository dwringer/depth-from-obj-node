from io import BytesIO

import matplotlib
matplotlib.use('AGG')
import numpy
from matplotlib.collections import PolyCollection
from PIL import Image

from invokeai.app.models.image import ImageCategory, ResourceOrigin
from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    InputField,
    invocation,
    InvocationContext,
    invocation_output,
)
from invokeai.app.invocations.primitives import (
    ImageField,
    ImageOutput
)


# Implemented as per https://matplotlib.org/matplotblog/posts/custom-3d-engine/

@invocation(
    "depth_map_from_wavefront_obj",
    title="Depth Map from Wavefront OBJ",
    tags=["image", "depth", "wavefront", "obj"],
    category="image",
    version="1.0.0",
)
class DepthMapFromWavefrontObjInvocation(BaseInvocation):
    """Renders a 3D depth map of a model described by a Wavefront .OBJ file"""
    width:  int = InputField(default=512, description="Width of the desired depth map output")
    height: int = InputField(default=512, description="Height of the desired depth map output")
    obj_file_path: str = InputField(default="", description="Path to a valid Wavefront .OBJ 3D model file")
    rotate_x: int = InputField(default=20, description="Degrees by which to tip the model about the x-axis after y-rotation")
    rotate_y: int = InputField(default=45, description="Degrees by which to rotate the model's bearing about the y-axis")
    translate_x: float = InputField(default=0, description="Normal units to translate camera in the x-axis")
    translate_y: float = InputField(default=0, description="Normal units to translate camera in the y-axis")
    translate_z: float = InputField(default=-3.5, description="Normal units to translate camera in the z-axis")
    fov: float = InputField(default=25, description="FOV for the camera viewport (in degrees)")

    def frustum(self, left, right, bottom, top, z_near, z_far):
        M = numpy.zeros((4, 4), dtype=numpy.float32)
        M[0, 0] = 2.0 * z_near / (right - left)
        M[1, 1] = 2.0 * z_near / (top - bottom)
        M[2, 2] = -(z_far + z_near) / (z_far - z_near)
        M[0, 2] = (right + left) / (right - left)
        M[2, 1] = (top + bottom) / (top - bottom)
        M[2, 3] = -2.0 * z_near * z_far / (z_far - z_near)
        M[3, 2] = -1.0
        return M

    def perspective(self, fov_x, aspect, z_near, z_far):
        width = numpy.tan(0.5 * numpy.radians(fov_x)) * z_near
        height  = width / aspect

        return self.frustum(-1 * width, width, -1 * height, height, z_near, z_far)
    
    def translate(self, x, y, z):
        return numpy.array([[1, 0, 0, x],
                            [0, 1, 0, y],
                            [0, 0, 1, z],
                            [0, 0, 0, 1]], dtype=float)

    def rotation_x(self, theta):
        theta_radians = numpy.pi * theta / 180
        cosine, sine = numpy.cos(theta_radians), numpy.sin(theta_radians)
        return numpy.array([[1, 0,       0,        0],
                            [0, cosine, -1 * sine, 0],
                            [0, sine,    cosine,   0],
                            [0, 0,       0,        1]], dtype=float)

    def rotation_y(self, theta):
        theta_radians = numpy.pi * theta / 180
        cosine, sine = numpy.cos(theta_radians), numpy.sin(theta_radians)
        return numpy.array([[cosine,    0, sine,   0],
                            [0,         1, 0,      0],
                            [-1 * sine, 0, cosine, 0],
                            [0,         0, 0,      1]], dtype=float)

    def invoke(self, context: InvocationContext) -> ImageOutput:
        vertices, faces = [], []

        # load the obj
        with open(self.obj_file_path) as inf:
            for line in inf.readlines():
                if line.startswith('#'):
                    continue
                values = line.split()
                if not values:
                    continue
                if values[0] == 'v':
                    vertices.append([float(x) for x in values[1:4]])
                elif values[0] == 'f':
                    faces.append([int(x.split('/')[0]) for x in values[1:4]])

        V, F = numpy.array(vertices), numpy.array(faces) - 1
            
        # normalize obj vertices between -1/2 and +1/2
        V = (V - (V.max(0) + V.min(0)) / 2) / max(V.max(0) - V.min(0))

        # create MVP matrix to position camera
        model = self.rotation_x(self.rotate_x) @ self.rotation_y(self.rotate_y)
        view  = self.translate(self.translate_x, self.translate_y, self.translate_z)
        projection = self.perspective(self.fov, self.width/self.height, 1, 100)
        MVP = projection @ view @ model

        # homogenize coordinates and do perspective projection
        V = numpy.c_[V, numpy.ones(len(V))] @ MVP.T

        # renormalize coordinates and prepare to render triangles
        V /= V[:, 3].reshape(-1, 1)
        V = V[F]
        
        # order triangles by average depth
        T = V[:, :, :2]
        Z = -V[:, :, 2].mean(axis=1)

        # prepare poly shading according to depth
        z_min, z_max = Z.min(), Z.max()
        Z = (Z - z_min) / (z_max - z_min)
        C = matplotlib.pyplot.get_cmap("gray")(Z)
        
        I = numpy.argsort(Z)
        T, C = T[I, :], C[I, :]

        # render
        inches_x, inches_y = 4, 4
        aspect = self.width/self.height
        xlim, ylim = [-1, 1], [-1, 1]
        figure = matplotlib.pyplot.figure(figsize=(inches_x, inches_y))
        axes = figure.add_axes([0, 0, 1, 1], xlim=xlim, ylim=ylim, aspect=1, frameon=False)
        collection = PolyCollection(T, closed=True, linewidth=0, facecolor=C, edgecolor='face')
        axes.add_collection(collection)

        # image_out needs to be a PIL image derived from our rendered matplotlib image
        image_buffer = BytesIO()
        matplotlib.pyplot.savefig(
            image_buffer, facecolor=(0,0,0,1), format='png', dpi=max(self.width, self.height)
        )
        image_out = Image.open(image_buffer)
        image_out = image_out.resize((self.width, self.height))
        
        image_dto = context.services.images.create(
            image=image_out,
            image_origin=ResourceOrigin.INTERNAL,
            image_category=ImageCategory.GENERAL,
            node_id=self.id,
            session_id=context.graph_execution_state_id,
            is_intermediate=self.is_intermediate
        )
        return ImageOutput(image=ImageField(image_name=image_dto.image_name),
                           width=image_dto.width,
                           height=image_dto.height,
        )
