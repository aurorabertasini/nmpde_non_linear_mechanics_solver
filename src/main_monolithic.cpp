#include "../include/SteadyNavierStokes.hpp"
#include "../include/MonolithicNavierStokes.hpp"
#include "../include/UncoupledNavierStokes.hpp"
#include "../include/ConfigReader.hpp"

int main(int argc, char *argv[])
{
    Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);
    const unsigned int mpi_rank =
        Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);

    int mpi_size;

    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    ConfigReader configReader;

    // Determine the mesh path from command line or config
    std::filesystem::path mesh2DPath;
    if (argc >= 2)
    {
        // Use the command-line argument as the mesh path
        mesh2DPath = argv[1];
    }
    else
    {
        // Fall back to the configuration file
        mesh2DPath = configReader.getMesh2DPath();
    }

    // The 3D mesh path is not used in this example but remains for completeness
    std::filesystem::path mesh3DPath = configReader.getMesh3DPath();

    int degreeVelocity = configReader.getDegreeVelocity();
    int degreePressure = configReader.getDegreePressure();
    double simulationPeriod = configReader.getSimulationPeriod();
    double timeStep = configReader.getTimeStep();
    double Re = configReader.getRe();

    auto start = std::chrono::high_resolution_clock::now();

    MonolithicNavierStokes<2> monolithicNavierStokes(mesh2DPath, degreeVelocity, degreePressure, simulationPeriod, timeStep, Re);
    monolithicNavierStokes.setup();
    monolithicNavierStokes.solve();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    if (mpi_rank == 0)
    {
        std::ofstream file_out("result_monolithic_mesh.csv", std::ios::app);
        if (!file_out)
        {
            std::cerr << "Error: Could not open result_uncoupled.csv for appending.\n";
            return 1;
        }

        file_out << mesh2DPath << "," << mpi_size << "," << elapsed.count() << std::endl;
        
    }
    return 0;
}