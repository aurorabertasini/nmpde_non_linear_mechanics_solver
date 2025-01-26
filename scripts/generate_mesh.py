import os
import subprocess as sp
from tqdm import tqdm

def generate_meshes():
    # Ricava la cartella in cui si trova questo script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Cartella con i file .geo, es: "scripts/mesh_scripts"
    mesh_scripts_dir = os.path.join(script_dir, 'mesh_scripts')

    # Cartella di destinazione per le .msh, "Mesh" al fianco di "scripts"
    mesh_output_dir = os.path.join(script_dir, '..', 'mesh')
    os.makedirs(mesh_output_dir, exist_ok=True)

    # Trova tutti i file .geo
    geo_files = [f for f in os.listdir(mesh_scripts_dir) if f.endswith('.geo')]
    if not geo_files:
        print("No .geo files found in mesh_scripts.")
        return

    # Formato personalizzato per togliere tempo e velocità
    # Mostrerà solo: "Generating meshes:  33%|████| 1/3"
    bar_format = "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}"

    # Creiamo la barra di avanzamento
    pbar = tqdm(
        geo_files,
        desc="Generating meshes",
        unit="file",
        bar_format=bar_format
    )

    for filename in pbar:
        # Determina se è 2D o 3D (in base a '2D' o '3D' nel nome file)
        if '2D' in filename:
            dimension = 2
        elif '3D' in filename:
            dimension = 3
        else:
            # Se non si può determinare, salta
            tqdm.write(f"Cannot determine dimension for {filename}, skipping.")
            continue
        
        # Aggiorniamo la descrizione dinamicamente (se vuoi mostrare il nome in corso)
        pbar.set_description_str(f"Processing: {filename} ({dimension}D)")
        
        geo_path = os.path.join(mesh_scripts_dir, filename)
        output_name = filename.replace('.geo', '.msh')
        output_path = os.path.join(mesh_output_dir, output_name)

        # Costruiamo il comando gmsh
        cmd = f'gmsh -{dimension} -format msh "{geo_path}" -o "{output_path}"'
        result = sp.run(cmd, shell=True, stdout=sp.DEVNULL, stderr=sp.STDOUT)

        if result.returncode != 0:
            tqdm.write(f"Error generating mesh for {filename}.")
        # else:
        #     tqdm.write(f"Mesh generated successfully: {output_name}")

    tqdm.write("All meshes have been generated successfully.")

if __name__ == '__main__':
    generate_meshes()
