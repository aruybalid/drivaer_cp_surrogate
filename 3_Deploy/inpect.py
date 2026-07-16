import pyvista as pv

mesh = pv.read("./references/run_2/boundary_2.vtp")
print("Cell data keys :", list(mesh.cell_data.keys()))
print("Point data keys:", list(mesh.point_data.keys()))
print("CpMeanTrim in cell_data?", 'CpMeanTrim' in mesh.cell_data)
print("CpMeanTrim in point_data?", 'CpMeanTrim' in mesh.point_data)