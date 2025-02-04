#include "../include/SteadyNavierStokes.hpp"
#include "../include/MonolithicNavierStokes.hpp"
#include "../include/UncoupledNavierStokes.hpp"
#include "../include/ConfigReader.hpp"

int main(int argc, char *argv[])
{
    Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);
    const unsigned int mpi_rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);

    int mpi_size; 
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    ConfigReader configReader;

    std::filesystem::path mesh2DPath      = configReader.getMesh2DPath();
    std::filesystem::path mesh3DPath      = configReader.getMesh3DPath();
    int degreeVelocity                    = configReader.getDegreeVelocity();
    int degreePressure                    = configReader.getDegreePressure();
    double simulationPeriod               = configReader.getSimulationPeriod();
    double timeStep                       = configReader.getTimeStep();
    double Re                             = configReader.getRe();

    auto start = std::chrono::high_resolution_clock::now();

    UncoupledNavierStokes<2> uncoupledNavierStokes(mesh2DPath,
                                                   degreeVelocity,
                                                   degreePressure,
                                                   simulationPeriod,
                                                   timeStep,
                                                   Re);
    uncoupledNavierStokes.run();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    if (mpi_rank == 0)
    {
        std::ofstream file_out("result_uncoupled.csv", std::ios::app);
        if (!file_out)
        {
            std::cerr << "Error: Could not open result_uncoupled.csv for appending.\n";
            return 1;
        }

        file_out << mpi_size << "," << elapsed.count() << std::endl;
        
    }

    return 0;
}
