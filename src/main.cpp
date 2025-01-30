#include "../include/Stokes.hpp"
#include "../include/Incremental_stokes.hpp"
#include "../include/MonolithicNavierStokes.hpp"
#include "../include/ChorinTemam.hpp"
#include "../include/IncrementalChorinTemam.hpp"
#include "../include/ConfigReader.hpp"

int main(int argc, char *argv[])
{
    Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);
    const unsigned int mpi_rank =
        Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);

    if (mpi_rank == 0)
    {
        std::cout << "Welcome to the Navier-Stokes solver" << std::endl;
    }

    double cylinder_radius = 0.1;

    double uM = 1.5;

    ConfigReader configReader;

    std::filesystem::path mesh2DPath = configReader.getMesh2DPath();
    std::filesystem::path mesh3DPath = configReader.getMesh3DPath();
    int degreeVelocity = configReader.getDegreeVelocity();
    int degreePressure = configReader.getDegreePressure();
    double simulationPeriod = configReader.getSimulationPeriod();
    double timeStep = configReader.getTimeStep();
    double Re = configReader.getRe();

    double nu = (uM * cylinder_radius) / Re;

    int choice = 0;

    if (mpi_rank == 0)
    { // std::cout << "Div u_star: " << div_u_star[q] << std::endl;

        std::cout << "Please choose the problem to solve:" << std::endl;
        std::cout << "(1) Convergence test 2D" << std::endl;
        std::cout << "(2) Convergence test 3D" << std::endl;
        std::cout << std::endl;
        std::cout << "Enter your choice: ";

        while (choice < 1 || choice > 2)
        {
            std::cin >> choice;
            if (choice < 1 || choice > 2)
            {
                std::cout << "Invalid choice. Please enter a valid choice: ";
            }
        }
    }

    MPI_Bcast(&choice, 1, MPI_INT, 0, MPI_COMM_WORLD);

    auto start = std::chrono::high_resolution_clock::now();

    double deltat_start = 0.1;
    double deltat = deltat_start;
    for (double i = 1.0; i <= 3; i += 1.0)
    {
        std::cout << deltat << std::endl;

        switch (choice)
        {
        case 1:
        {
            IncrementalChorinTemam<2> incrementalChorinTemam(mesh2DPath, degreeVelocity, degreePressure, simulationPeriod, deltat, Re);
            incrementalChorinTemam.run();
            double error_norm = incrementalChorinTemam.compute_error(VectorTools::L2_norm);
            // write to file
            std::ofstream out_file("error_norm_2d.csv", std::ios::app);
            out_file << deltat << ","
                     << error_norm << "\n";
            out_file.close();
            break;
        }
        case 2:
        {
            IncrementalChorinTemam<3> incrementalChorinTemam(mesh3DPath, degreeVelocity, degreePressure, simulationPeriod, deltat, Re);
            incrementalChorinTemam.run();
            double error_norm = incrementalChorinTemam.compute_error(VectorTools::L2_norm);
            // write to file
            std::ofstream out_file("error_norm_3d.csv", std::ios::app);
            out_file << deltat << ","
                     << error_norm << "\n";
            out_file.close();
            break;
        }
            return 0;
        }

        deltat /= 2.0;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    if (mpi_rank == 0)
    {
        std::cout << "Elapsed time: " << elapsed.count() << " s\n";
        std::cout << std::endl
                  << "THE END" << std::endl;
    }
}
