#include "../include/MonolithicNavierStokes.hpp"
#include "../include/UncoupledNavierStokes.hpp"
#include "../include/SteadyNavierStokes.hpp"

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <iomanip>

void writeErrorsToFile(const std::string& filename,
                        const std::vector<double>& deltat_vector,
                        const std::vector<double>& errors_Linf_pressure,
                        const std::vector<double>& errors_L2_velocity,
                        const std::vector<double>& errors_H1_velocity)
{
    std::ofstream outFile(filename);
    if (!outFile)
    {
        std::cerr << "Error: Unable to open file " << filename << " for writing." << std::endl;
        return;
    }

    for (size_t i = 0; i < deltat_vector.size(); ++i)
    {
        outFile << std::scientific << "dt = " << std::setw(4)
                << std::setprecision(2) << deltat_vector[i];

        outFile << std::scientific << " | pLinf = " << errors_Linf_pressure[i];
        if (i > 0)
        {
            double p = std::log(errors_Linf_pressure[i] / errors_Linf_pressure[i - 1]) /
                       std::log(deltat_vector[i] / deltat_vector[i - 1]);
            outFile << " (" << std::fixed << std::setprecision(2)
                    << std::setw(4) << p << ")";
        }
        else
        {
            outFile << " (  - )";
        }

        outFile << std::scientific << " | vL2 = " << errors_L2_velocity[i];
        if (i > 0)
        {
            double p = std::log(errors_L2_velocity[i] / errors_L2_velocity[i - 1]) /
                       std::log(deltat_vector[i] / deltat_vector[i - 1]);
            outFile << " (" << std::fixed << std::setprecision(2)
                    << std::setw(4) << p << ")";
        }
        else
        {
            outFile << " (  - )";
        }

        outFile << std::scientific << " | vH1 = " << errors_H1_velocity[i];
        if (i > 0)
        {
            double p = std::log(errors_H1_velocity[i] / errors_H1_velocity[i - 1]) /
                       std::log(deltat_vector[i] / deltat_vector[i - 1]);
            outFile << " (" << std::fixed << std::setprecision(2)
                    << std::setw(4) << p << ")";
        }
        else
        {
            outFile << " (  - )";
        }

        outFile << "\n";
    }

    outFile.close();
}


int main(int argc, char *argv[])
{
    Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);
    const unsigned int mpi_rank =
        Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);

    if (mpi_rank == 0)
    {
        std::cout << "Welcome to the Numerical Analysis brench of the Navier-Stokes solver" << std::endl;
    }

    std::filesystem::path mesh2DPath = "../mesh/Square2D.msh";
    std::filesystem::path mesh3DPath = "../mesh/Cube3D.msh";
    int degreeVelocity = 2;
    int degreePressure = 1;
    double simulationPeriod = 1.0;

    int choice = 0;

    if (mpi_rank == 0)
    { // std::cout << "Div u_star: " << div_u_star[q] << std::endl;

        std::cout << "Please choose the problem to solve:" << std::endl;
        std::cout << "(1) Time Convergence test ChorinTemam (2D)" << std::endl;
        std::cout << "(2) Space Convergence test ChorinTemam (3D)" << std::endl;
        std::cout << "(3) Space Convergence test ChorinTemam (2D)" << std::endl;
        std::cout << "(4) Space Convergence test SteadyNavierStokes (2D)" << std::endl;
        std::cout << std::endl;
        std::cout << "Enter your choice: ";

        while (choice < 1 || choice > 4)
        {
            std::cin >> choice;
            if (choice < 1 || choice > 4)
            {
                std::cout << "Invalid choice. Please enter a valid choice: ";
            }
        }
    }

    MPI_Bcast(&choice, 1, MPI_INT, 0, MPI_COMM_WORLD);

    auto start = std::chrono::high_resolution_clock::now();

    switch (choice)
    {
    case 1:
    {
        std::ofstream out_file("time_convergence_analysis_chorin_temam.csv", std::ios::app);

        out_file << "delta_t,pressure_Linf_error,velocity_L2_error,velocity_H1_error" << std::endl;

        std::vector<double> errors_Linf_pressure;
        std::vector<double> errors_H1_velocity;
        std::vector<double> errors_L2_velocity;
        std::vector<double> deltat_vector;

        double deltat_start = 0.1;
        double deltat = deltat_start;
        int number_of_delta_t_refinements = 5;
        int count = 0;
        do
        {
            deltat_vector.push_back(deltat);
            UncoupledNavierStokes<2> uncoupledNavierStokes(mesh2DPath, degreeVelocity, degreePressure, simulationPeriod, deltat);
            uncoupledNavierStokes.run();

            double pressure_Linf_error = uncoupledNavierStokes.compute_error_pressure(VectorTools::Linfty_norm);
            double velocity_L2_error = uncoupledNavierStokes.compute_error_velocity(VectorTools::L2_norm);
            double velocity_H1_error = uncoupledNavierStokes.compute_error_velocity(VectorTools::H1_norm);

            if (mpi_rank == 0)
            {
                out_file << deltat << "," << pressure_Linf_error << "," << velocity_L2_error << "," << velocity_H1_error << std::endl;

                errors_Linf_pressure.push_back(pressure_Linf_error);
                errors_L2_velocity.push_back(velocity_L2_error);
                errors_H1_velocity.push_back(velocity_H1_error);
            }

            deltat /= 2.0;
            count++;
        } while (count < number_of_delta_t_refinements);

        writeErrorsToFile("time_convergence_analysis_chorin_temam_" + std::to_string(simulationPeriod) + ".txt", deltat_vector, errors_Linf_pressure, errors_L2_velocity, errors_H1_velocity);

        break;
    }
    case 2:
    {
        std::vector<std::string> meshFiles = {"../mesh/cubeBenchmark3D_1.msh", "../mesh/cubeBenchmark3D_2.msh", "../mesh/cubeBenchmark3D_3.msh", "../mesh/cubeBenchmark3D_4.msh"};
        std::ofstream out_file("space_convergence_analysis_chorin_temam_3D.csv", std::ios::app);
        out_file << "mesh_file_name,pressure_Linf_error,velocity_H1_error" << std::endl;

        double number_of_time_steps = 400.0;
        double deltat = 1e-7;

        for (auto meshFile : meshFiles)
        {
            UncoupledNavierStokes<3> uncoupledNavierStokes(meshFile, degreeVelocity, degreePressure, deltat * number_of_time_steps, deltat);
            uncoupledNavierStokes.run();

            double pressure_Linf_error = uncoupledNavierStokes.compute_error_pressure(VectorTools::L2_norm);
            double velocity_H1_error = uncoupledNavierStokes.compute_error_velocity(VectorTools::H1_norm);

            if (mpi_rank == 0)
            {
                out_file << meshFile << "," << pressure_Linf_error << "," << velocity_H1_error << std::endl;
            }
        }
        break;
    }
    case 3:
    {
        std::vector<std::string> meshFiles = {"../mesh/squareBenchmark2D_1000.msh", "../mesh/squareBenchmark2D_500.msh", "../mesh/squareBenchmark2D_250.msh", "../mesh/squareBenchmark2D_125.msh"};
        std::ofstream out_file("space_convergence_analysis_chorin_temam_2D.csv", std::ios::app);
        out_file << "mesh_file_name,pressure_Linf_error,velocity_H1_error" << std::endl;
        double number_of_time_steps = 4.0;
        double deltat = 1e-10;

        for (auto meshFile : meshFiles)
        {
            UncoupledNavierStokes<2> uncoupledNavierStokes(meshFile, degreeVelocity, degreePressure, deltat * number_of_time_steps, deltat);
            uncoupledNavierStokes.run();

            double pressure_Linf_error = uncoupledNavierStokes.compute_error_pressure(VectorTools::L2_norm);
            double velocity_H1_error = uncoupledNavierStokes.compute_error_velocity(VectorTools::H1_norm);

            if (mpi_rank == 0)
            {
                out_file << meshFile << "," << pressure_Linf_error << "," << velocity_H1_error << std::endl;
            }
        }
    }

    case 4:
    {
        std::vector<std::string> meshFiles = {"../mesh/squareBenchmark2D_1000.msh", "../mesh/squareBenchmark2D_500.msh", "../mesh/squareBenchmark2D_250.msh", "../mesh/squareBenchmark2D_125.msh"};
        std::ofstream out_file("space_convergence_analysis_steady_navier_stokes_2D.csv", std::ios::app);
        out_file << "mesh_file_name,velocity_L2_error,velocity_H1_error" << std::endl;


        for (auto meshFile : meshFiles)
        {
            double velocity_L2_error = 0.0;
            double velocity_H1_error = 0.0;
            SteadyNavierStokes<2> steadyNavierStokes(meshFile, degreeVelocity, degreePressure);
            steadyNavierStokes.run_full_problem_pipeline(velocity_L2_error, velocity_H1_error);

            if (mpi_rank == 0)
            {
                out_file << meshFile << "," << velocity_L2_error << "," << velocity_H1_error << std::endl;
            }
        }
    }

  }
}
