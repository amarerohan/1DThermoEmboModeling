import vtk
import numpy as np
import pandas as pd
# from find_IndependentTrees import findTrees
import time


def read_vtk_polydata(filename):
    # Create a reader for ASCII VTK files
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(filename)
    reader.Update()

    # Get the output polydata
    polydata = reader.GetOutput()

    # Extract points
    points = polydata.GetPoints()
    num_points = points.GetNumberOfPoints()
    point_coordinates = [points.GetPoint(i) for i in range(num_points)]

    # Extract lines
    lines = polydata.GetLines()
    num_lines = lines.GetNumberOfCells()
    line_data = []
    lines.InitTraversal()
    idList = vtk.vtkIdList()
    while lines.GetNextCell(idList):
        line_points = [idList.GetId(j) for j in range(idList.GetNumberOfIds())]
        line_data.append(line_points)

    # Extract point data (if available)
    point_data = polydata.GetPointData()
    
    # Example: Extract radius data (if available)
    radius_array = point_data.GetArray("Radius")
    radius_data = []
    if radius_array:
        for i in range(num_points):
            radius_data.append(radius_array.GetValue(i))
    
    # Example: Extract damage data (if available)
    damage_array = point_data.GetArray("Damage")
    damage_data = []
    if damage_array:
        for i in range(num_points):
            damage_data.append(damage_array.GetValue(i))
            
    return damage_data, point_coordinates
    # return point_coordinates, line_data, radius_data, damage_data



