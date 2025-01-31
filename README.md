### Generating the mesh
The mesh is generated using the `gmsh` software. To generate all the meshes with the same command run the python script `generate_mesh.py` in the folder scripts.
For instance, if you are located in the root folder of the project, you can run the following command:
```bash
python scripts/generate_mesh.py
``` 

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