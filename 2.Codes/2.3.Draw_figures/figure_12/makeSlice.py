from paraview.simple import *
import sys

file_vtk = XMLRectilinearGridReader(FileName=[sys.argv[1]])
I3_Slice = Slice(Input = file_vtk)
I3_Slice.SliceType.Origin = [float(sys.argv[3])/2.0, -330, -float(sys.argv[3])/2.0]
I3_Slice.SliceType.Normal = [0.0, 1.0, 0.0]
SaveData(sys.argv[2], proxy=I3_Slice)
