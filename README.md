### Generating the mesh
The mesh is generated using the `gmsh` software. To generate all the meshes with the same command run the python script `generate_mesh.py` in the folder scripts.
For instance, if you are located in the root folder of the project, you can run the following command:
```bash
python scripts/generate_mesh.py
``` 
### Modify the parameters of the simulation
The parameters of the simulation can be modified in the file `parameters.config`. The file presents a custom format. Lines that start with `#` are considered comments and are ignored. By this file the following parameters can be modified:
- `mesh_2d_path`: path to the 2D mesh file
- `mesh_3d_path`: path to the 3D mesh file
- `degree_velocity`: degree of the velocity space
- `degree_pressure`: degree of the pressure space
- `T`: final time of the simulation
- `deltat`: time step
- `Re`: Reynolds number

If one of these parameters is not present in the file, it will be asked to the user at the beginning of the simulation.

### Compiling
To build the executable, make sure you have loaded the needed modules with
```bash
$ module load gcc-glibc dealii
```
Then run the following commands:
```bash
$ mkdir build
$ cd build
$ cmake ..
$ make
```
The executable will be created into `build`, and can be executed through
```bash
$ ./executable-name
```