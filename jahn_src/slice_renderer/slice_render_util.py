
import os
import vtkmodules.vtkInteractionStyle
# noinspection PyUnresolvedReferences
import vtkmodules.vtkRenderingOpenGL2
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkFiltersSources import vtkSphereSource
from vtkmodules.vtkIOImage import (
    vtkBMPWriter,
    vtkJPEGWriter,
    vtkPNGWriter,
    vtkPNMWriter,
    vtkPostScriptWriter,
    vtkTIFFWriter
)
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkPolyDataMapper,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkRenderer,
    vtkWindowToImageFilter
)
import numpy as np

def WriteImage(fileName, renWin, rgba=True):
    '''
    Write the render window view to an image file.

    Image types supported are:
     BMP, JPEG, PNM, PNG, PostScript, TIFF.
    The default parameters are used for all writers, change as needed.

    :param fileName: The file name, if no extension then PNG is assumed.
    :param renWin: The render window.
    :param rgba: Used to set the buffer type.
    :return:
    '''

    import os

    if fileName:
        # Select the writer to use.
        path, ext = os.path.splitext(fileName)
        ext = ext.lower()
        if not ext:
            ext = '.png'
            fileName = fileName + ext
        if ext == '.bmp':
            writer = vtkBMPWriter()
        elif ext == '.jpg':
            writer = vtkJPEGWriter()
        elif ext == '.pnm':
            writer = vtkPNMWriter()
        elif ext == '.ps':
            if rgba:
                rgba = False
            writer = vtkPostScriptWriter()
        elif ext == '.tiff':
            writer = vtkTIFFWriter()
        else:
            writer = vtkPNGWriter()

        windowto_image_filter = vtkWindowToImageFilter()
        windowto_image_filter.SetInput(renWin)
        windowto_image_filter.SetScale(1)  # image quality
        if rgba:
            windowto_image_filter.SetInputBufferTypeToRGBA()
        else:
            windowto_image_filter.SetInputBufferTypeToRGB()
            # Read from the front buffer.
            windowto_image_filter.ReadFrontBufferOff()
            windowto_image_filter.Update()

        writer.SetFileName(fileName)
        writer.SetInputConnection(windowto_image_filter.GetOutputPort())
        writer.Write()
    else:
        raise RuntimeError('Need a filename.')


def concat_objs(_dir, output_filename):
    #_dir = '/data/jhahn/data/shape_dataset/data/shape/vase/0/fractured_0'
    #_dir = '/disk2/data/shape_dataset/data/shape/cube/1/cube_32_16_0_1_0'
    files = [_dir+"/"+f for f in os.listdir(_dir) if os.path.isfile(_dir+"/"+f)]

    #rot_mat = scipy.spatial.transform.Rotation.from_rotvec(np.pi/2 * np.array([0, 0, 1])).as_matrix()

    f_2_last = []
    with open(output_filename, 'w') as outfile:

        f_2_last.append(0)
        for fname in files:
            
            _c = 0
            with open(fname) as infile:
                _pcs = []
                for line in infile:
                    if line.lower().startswith('v'):
                        _c += 1
                        _arr = line[2:].split()
                        _arr = np.array([float(a) for a in _arr])                    
                        _pcs.append(_arr)

                _pcs = np.array(_pcs)
                #normal_vector = calculate_normal_vector_open3d(_pcs)
                #_pcs = rotate_and_translate_to_xy_plane(_pcs, normal_vector)
                for _arr in _pcs:
                    outfile.write(f'v {_arr[0]} {_arr[1]} {_arr[2]}\n')
                    
            

            f_2_last.append(_c)
        print(f_2_last)
        _delta = 0
        for fi, fname in enumerate(files):
            _delta += f_2_last[fi]
            with open(fname) as infile:
                for line in infile:
                    if line.lower().startswith('f'):
                        _arr = line[2:].split()
                        outfile.write(f'f {int(_arr[0])+_delta} {int(_arr[1])+_delta} {int(_arr[2])+_delta}\n')