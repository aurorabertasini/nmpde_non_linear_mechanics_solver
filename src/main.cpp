#include "../include/MonolithicNavierStokes.hpp"
#include "../include/UncoupledNavierStokes.hpp"
#include "../include/SteadyNavierStokes.hpp"

#include <filesystem>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <chrono>

// -----------------------------------------------------------------------------
// 1) TIME ERROR PRINTOUT 
// -----------------------------------------------------------------------------
void writeTimeErrorsToFile(const std::string &filename,
                           const std::vector<double> &deltat_vector,
                           const std::vector<double> &errors_pLinf,
                           const std::vector<double> &errors_vL2,
                           const std::vector<double> &errors_vH1,
                           const std::vector<double> &errors_vLinfinityH1,
                           const std::vector<double> &errors_pLinfinityL2,
                           const std::vector<double> &errors_vLinfinityL2)
{
    std::ofstream outFile(filename);
    if (!outFile)
    {
        std::cerr << "Error: Unable to open file " << filename << " for writing." << std::endl;
        return;
    }

    // Loop over each time step size
    for (size_t i = 0; i < deltat_vector.size(); ++i)
    {
        // dt
        outFile << std::scientific
                << "dt = " << std::setw(8) << std::setprecision(4)
                << deltat_vector[i];

        // 1) Pressure Linf
        outFile << std::scientific
                << " | pLinf = " << errors_pLinf[i];
        if (i > 0)
        {
            double p = std::log(errors_pLinf[i] / errors_pLinf[i - 1]) /
                       std::log(deltat_vector[i] / deltat_vector[i - 1]);
            outFile << " (" << std::fixed << std::setprecision(2) << p << ")";
        }
        else
        {
            outFile << " (  - )";
        }

        // 2) Velocity L2
        outFile << std::scientific
                << " | vL2 = " << errors_vL2[i];
        if (i > 0)
        {
            double p = std::log(errors_vL2[i] / errors_vL2[i - 1]) /
                       std::log(deltat_vector[i] / deltat_vector[i - 1]);
            outFile << " (" << std::fixed << std::setprecision(2) << p << ")";
        }
        else
        {
            outFile << " (  - )";
        }

        // 3) Velocity H1
        outFile << std::scientific
                << " | vH1 = " << errors_vH1[i];
        if (i > 0)
        {
            double p = std::log(errors_vH1[i] / errors_vH1[i - 1]) /
                       std::log(deltat_vector[i] / deltat_vector[i - 1]);
            outFile << " (" << std::fixed << std::setprecision(2) << p << ")";
        }
        else
        {
            outFile << " (  - )";
        }

        // 4) Velocity Linfinity H1
        outFile << std::scientific
                << " | vLinfinityH1 = " << errors_vLinfinityH1[i];
        if (i > 0)
        {
            double p = std::log(errors_vLinfinityH1[i] / errors_vLinfinityH1[i - 1]) /
                       std::log(deltat_vector[i] / deltat_vector[i - 1]);
            outFile << " (" << std::fixed << std::setprecision(2) << p << ")";
        }
        else
        {
            outFile << " (  - )";
        }

        // 5) Pressure Linfinity L2
        outFile << std::scientific
                << " | pLinfinityL2 = " << errors_pLinfinityL2[i];
        if (i > 0)
        {
            double p = std::log(errors_pLinfinityL2[i] / errors_pLinfinityL2[i - 1]) /
                       std::log(deltat_vector[i] / deltat_vector[i - 1]);
            outFile << " (" << std::fixed << std::setprecision(2) << p << ")";
        }
        else
        {
            outFile << " (  - )";
        }

        // 6) Velocity Linfinity L2
        outFile << std::scientific
                << " | vLinfinityL2 = " << errors_vLinfinityL2[i];
        if (i > 0)
        {
            double p = std::log(errors_vLinfinityL2[i] / errors_vLinfinityL2[i - 1]) /
                       std::log(deltat_vector[i] / deltat_vector[i - 1]);
            outFile << " (" << std::fixed << std::setprecision(2) << p << ")";
        }
        else
        {
            outFile << " (  - )";
        }

        outFile << "\n";
    }

    outFile.close();
}

// -----------------------------------------------------------------------------
// 2) SPACE ERROR PRINTOUT 
// -----------------------------------------------------------------------------
void writeSpaceErrorsToFile(const std::string &filename,
                            const std::vector<double> &mesh_sizes,
                            const std::vector<double> &errors_Linf_pressure,
                            const std::vector<double> &errors_H1_velocity)
{
    std::ofstream outFile(filename);
    if (!outFile)
    {
        std::cerr << "Error: Unable to open file " << filename << " for writing." << std::endl;
        return;
    }

    for (size_t i = 0; i < mesh_sizes.size(); ++i)
    {
        // Mesh size
        outFile << std::scientific << "h = " << std::setw(8)
                << std::setprecision(4) << mesh_sizes[i];

        // p Linf
        outFile << " | pLinf = " << errors_Linf_pressure[i];
        if (i > 0)
        {
            double p = std::log(errors_Linf_pressure[i] / errors_Linf_pressure[i - 1]) /
                       std::log(mesh_sizes[i] / mesh_sizes[i - 1]);
            outFile << " (" << std::fixed << std::setprecision(2)
                    << std::setw(4) << p << ")";
        }
        else
        {
            outFile << " (  - )";
        }

        // v H1
        outFile << std::scientific << " | vH1 = " << errors_H1_velocity[i];
        if (i > 0)
        {
            double p = std::log(errors_H1_velocity[i] / errors_H1_velocity[i - 1]) /
                       std::log(mesh_sizes[i] / mesh_sizes[i - 1]);
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

void writeSpaceErrorsForSteadyToFile(const std::string &filename,
                                     const std::vector<double> &mesh_sizes,
                                     const std::vector<double> &errors_vL2,
                                     const std::vector<double> &errors_vH1)
{
    std::ofstream outFile(filename);
    if (!outFile)
    {
        std::cerr << "Error: Unable to open file " << filename << " for writing." << std::endl;
        return;
    }

    for (size_t i = 0; i < mesh_sizes.size(); ++i)
    {
        // Mesh size
        outFile << std::scientific << "h = " << std::setw(8)
                << std::setprecision(4) << mesh_sizes[i];

        // Velocity L2 error
        outFile << " | vL2 = " << errors_vL2[i];
        if (i > 0)
        {
            double p = std::log(errors_vL2[i] / errors_vL2[i - 1]) /
                       std::log(mesh_sizes[i] / mesh_sizes[i - 1]);
            outFile << " (" << std::fixed << std::setprecision(2) << p << ")";
        }
        else
        {
            outFile << " (  - )";
        }

        // Velocity H1 error
        outFile << " | vH1 = " << errors_vH1[i];
        if (i > 0)
        {
            double p = std::log(errors_vH1[i] / errors_vH1[i - 1]) /
                       std::log(mesh_sizes[i] / mesh_sizes[i - 1]);
            outFile << " (" << std::fixed << std::setprecision(2) << p << ")";
        }
        else
        {
            outFile << " (  - )";
        }

        outFile << "\n";
    }

    outFile.close();
}

// -----------------------------------------------------------------------------
// MAIN
// -----------------------------------------------------------------------------
int main(int argc, char *argv[])
{
    Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);
    const unsigned int mpi_rank =
        Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);

    if (mpi_rank == 0)
    {
        std::cout << "Welcome to the Numerical Analysis branch of the Navier-Stokes solver" << std::endl;
    }

    std::filesystem::path mesh2DPath = "../mesh/Square2D.msh";
    std::filesystem::path mesh3DPath = "../mesh/Cube3D.msh";
    int degreeVelocity = 2;
    int degreePressure = 1;
    double simulationPeriod = 1.0;

    int choice = 0;

    if (mpi_rank == 0)
    {
        std::cout << "Please choose the problem to solve:\n"
                  << "(1) Time Convergence test ChorinTemam (2D)\n"
                  << "(2) Space Convergence test Monolithic (3D)\n"
                  << "(3) Space Convergence test SteadyNavierStokes (2D)\n\n"
                  << "Enter your choice: ";

        while (choice < 1 || choice > 4)
        {
            std::cin >> choice;
            if (choice < 1 || choice > 4)
            {
                std::cout << "Invalid choice. Please enter a valid choice: ";
            }
        }
    }

    // Broadcast choice to all MPI processes
    MPI_Bcast(&choice, 1, MPI_INT, 0, MPI_COMM_WORLD);

    auto start = std::chrono::high_resolution_clock::now();

    switch (choice)
    {
    // -------------------------------------------------------------------------
    // CASE 1: Time Convergence test Chorin-Temam (2D)
    // -------------------------------------------------------------------------
    case 1:
    {
        // We append (or overwrite) the CSV file
        std::ofstream out_file("time_convergence_analysis_chorin_temam.csv", std::ios::app);

        // Write CSV header (do it once if needed, or check file size, etc.)
        if (mpi_rank == 0)
        {
            out_file << "delta_t,"
                     << "pLinf,"
                     << "vL2,"
                     << "vH1,"
                     << "vLinfinityH1,"
                     << "pLinfinityL2,"
                     << "vLinfinityL2\n";
        }

        // Vectors to store the errors for EOC
        std::vector<double> deltat_vector;
        std::vector<double> errors_pLinf;
        std::vector<double> errors_vL2;
        std::vector<double> errors_vH1;
        std::vector<double> errors_vLinfinityH1;
        std::vector<double> errors_pLinfinityL2;
        std::vector<double> errors_vLinfinityL2;

        double deltat_start = 0.1;
        double deltat = deltat_start;
        int number_of_delta_t_refinements = 5;
        int count = 0;
        do
        {
            deltat_vector.push_back(deltat);

            // Run the Uncoupled NavierStokes solver
            UncoupledNavierStokes<2> uncoupledNavierStokes(
                mesh2DPath,
                degreeVelocity,
                degreePressure,
                simulationPeriod,
                deltat);

            uncoupledNavierStokes.run();

            // Retrieve the various errors
            double pLinf = uncoupledNavierStokes.get_Linfinity_error_pressure();
            double vL2 = uncoupledNavierStokes.get_L2_error_velocity();
            double vH1 = uncoupledNavierStokes.get_H1_error_velocity();
            double vLinfinityH1 = uncoupledNavierStokes.get_linfinity_H1_error_velocity();
            double pLinfinityL2 = uncoupledNavierStokes.get_linfinity_L2_error_pressure();
            double vLinfinityL2 = uncoupledNavierStokes.get_linfinity_L2_error_velocity();

            // Collect in vectors for EOC
            if (mpi_rank == 0)
            {
                errors_pLinf.push_back(pLinf);
                errors_vL2.push_back(vL2);
                errors_vH1.push_back(vH1);
                errors_vLinfinityH1.push_back(vLinfinityH1);
                errors_pLinfinityL2.push_back(pLinfinityL2);
                errors_vLinfinityL2.push_back(vLinfinityL2);

                // Write to CSV
                out_file << deltat << ","
                         << pLinf << ","
                         << vL2 << ","
                         << vH1 << ","
                         << vLinfinityH1 << ","
                         << pLinfinityL2 << ","
                         << vLinfinityL2 << "\n";
            }

            deltat /= 2.0;
            count++;
        } while (count < number_of_delta_t_refinements);

        out_file.close();

        // Write extended time convergence results (with EOC)
        if (mpi_rank == 0)
        {
            writeTimeErrorsToFile(
                "time_convergence_analysis_chorin_temam_" + std::to_string(simulationPeriod) + ".txt",
                deltat_vector,
                errors_pLinf,
                errors_vL2,
                errors_vH1,
                errors_vLinfinityH1,
                errors_pLinfinityL2,
                errors_vLinfinityL2);
        }
        break;
    }
    
    // -------------------------------------------------------------------------
    // CASE 2: Space Convergence test Monolithic (3D)
    // -------------------------------------------------------------------------
    case 2:
    {
        // Different mesh files with known characteristic size h:
        std::vector<std::string> meshFiles = {
            "../mesh/Cube3D_1.msh",
            "../mesh/Cube3D_2.msh",
            "../mesh/Cube3D_3.msh",
            "../mesh/Cube3D_4.msh"};
        // Example mesh sizes (must correspond to the actual meshes above):
        std::vector<double> mesh_sizes = {0.5, 0.25, 0.125, 0.0625};

        // Vectors to store errors for each mesh
        std::vector<double> errors_Linf_pressure;
        std::vector<double> errors_H1_velocity;

        // For a "space convergence" test, we typically fix time to something
        // small or do a steady run, but let's pick final_time and delta_t
        double number_of_timesteps = 4.0;
        double delta_t = 1e-6;

        // Output CSV for quick reference
        std::ofstream out_file("space_convergence_analysis_monolithic_3D.csv", std::ios::app);
        if (mpi_rank == 0)
            out_file << "mesh_file,h,pLinf,vH1\n";

        // Loop over each mesh
        for (size_t i = 0; i < meshFiles.size(); ++i)
        {
            MonolithicNavierStokes<3> monolithicNavierStokes(
                meshFiles[i],
                degreeVelocity,
                degreePressure,
                number_of_timesteps * delta_t,
                delta_t);

            monolithicNavierStokes.setup();
            monolithicNavierStokes.solve();

            double pLinf_error = monolithicNavierStokes.compute_error(VectorTools::Linfty_norm, /*velocity=*/false);
            double vH1_error = monolithicNavierStokes.compute_error(VectorTools::H1_norm, /*velocity=*/true);

            // Collect results
            errors_Linf_pressure.push_back(pLinf_error);
            errors_H1_velocity.push_back(vH1_error);

            if (mpi_rank == 0)
            {
                out_file << meshFiles[i] << ","
                         << mesh_sizes[i] << ","
                         << pLinf_error << ","
                         << vH1_error << std::endl;
            }
        }

        if (mpi_rank == 0)
        {
            writeSpaceErrorsToFile("space_convergence_analysis_monolithic_3D.txt",
                                   mesh_sizes,
                                   errors_Linf_pressure,
                                   errors_H1_velocity);
        }
        break;
    }

    // -------------------------------------------------------------------------
    // CASE 3: Space Convergence test SteadyNavierStokes (2D)
    //         
    // -------------------------------------------------------------------------
    case 3:
    {
        // Example 2D meshes
        std::vector<std::string> meshFiles = 
        {
            "../mesh/Square2D_1.msh",
            "../mesh/Square2D_2.msh",
            "../mesh/Square2D_3.msh"
        };
        // Hypothetical characteristic mesh sizes
        std::vector<double> mesh_sizes = {0.1, 0.05, 0.025};

        // Vectors to store velocity errors for each mesh
        std::vector<double> errors_vL2;
        std::vector<double> errors_vH1;

        // We also open a CSV for quick reference
        std::ofstream out_file("space_convergence_analysis_steady_navier_stokes_2D.csv", std::ios::app);
        if (mpi_rank == 0)
            out_file << "mesh_file_name,velocity_L2_error,velocity_H1_error\n";

        for (size_t i = 0; i < meshFiles.size(); ++i)
        {
            double velocity_L2_error = 0.0;
            double velocity_H1_error = 0.0;

            // Construct and run the steady Navier Stokes problem
            SteadyNavierStokes<2> steadyNavierStokes(
                meshFiles[i],
                degreeVelocity,
                degreePressure);

            // This function presumably solves the problem and fills the references
            steadyNavierStokes.run_full_problem_pipeline(velocity_L2_error, velocity_H1_error);

            // Collect the computed velocity errors
            errors_vL2.push_back(velocity_L2_error);
            errors_vH1.push_back(velocity_H1_error);

            if (mpi_rank == 0)
            {
                out_file << meshFiles[i] << ","
                         << velocity_L2_error << ","
                         << velocity_H1_error << "\n";
            }
        }

        // Write space convergence results (with EOC) using velocity L2 and velocity H1 errors
        if (mpi_rank == 0)
        {
            writeSpaceErrorsForSteadyToFile("space_convergence_analysis_steady_navier_stokes_2D.txt",
                                            mesh_sizes,
                                            errors_vL2,
                                            errors_vH1);
        }
        break;
    }

    default:
        break;
    }

    // -------------------------------------------------------------------------
    // End timing
    // -------------------------------------------------------------------------
    auto end = std::chrono::high_resolution_clock::now();
    if (mpi_rank == 0)
    {
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
        std::cout << "Computation finished. Elapsed time: " << elapsed << " seconds.\n";
    }

    return 0;
}
