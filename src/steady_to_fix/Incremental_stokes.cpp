#include "../../include/Incremental_stokes.hpp"

void IncrementalStokes::setup()
{
    Linardo::setup();

    // Initialize the finite element space.
    {
        pcout << "Initializing the finite element space" << std::endl;

        const FE_SimplexP<dim> fe_scalar_velocity(degree_velocity);
        const FE_SimplexP<dim> fe_scalar_pressure(degree_pressure);
        fe = std::make_unique<FESystem<dim>>(fe_scalar_velocity, dim, fe_scalar_pressure, 1);

        pcout << "  Velocity degree:           = " << fe_scalar_velocity.degree << std::endl;
        pcout << "  Pressure degree:           = " << fe_scalar_pressure.degree << std::endl;
        pcout << "  DoFs per cell              = " << fe->dofs_per_cell << std::endl;

        quadrature = std::make_unique<QGaussSimplex<dim>>(fe->degree + 1);

        pcout << "  Quadrature points per cell = " << quadrature->size() << std::endl;

        quadrature_face = std::make_unique<QGaussSimplex<dim - 1>>(fe->degree + 1);

        pcout << "  Quadrature points per face = " << quadrature_face->size() << std::endl;
    }

    pcout << "-----------------------------------------------" << std::endl;

    // Initialize the DoF handler.
    {
        pcout << "Initializing the DoF handler" << std::endl;

        dof_handler.reinit(mesh);
        dof_handler.distribute_dofs(*fe);

        // We want to reorder DoFs so that all velocity DoFs come first, and then
        // all pressure DoFs.
        std::vector<unsigned int> block_component(dim + 1, 0);
        block_component[dim] = 1;
        DoFRenumbering::component_wise(dof_handler, block_component);

        locally_owned_dofs = dof_handler.locally_owned_dofs();
        DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

        std::vector<types::global_dof_index> dofs_per_block =
            DoFTools::count_dofs_per_fe_block(dof_handler, block_component);
        const unsigned int n_u = dofs_per_block[0];
        const unsigned int n_p = dofs_per_block[1];

        block_owned_dofs.resize(2);
        block_relevant_dofs.resize(2);
        block_owned_dofs[0] = locally_owned_dofs.get_view(0, n_u);
        block_owned_dofs[1] = locally_owned_dofs.get_view(n_u, n_u + n_p);
        block_relevant_dofs[0] = locally_relevant_dofs.get_view(0, n_u);
        block_relevant_dofs[1] = locally_relevant_dofs.get_view(n_u, n_u + n_p);

        pcout << "  Number of DoFs: " << std::endl;
        pcout << "    velocity = " << n_u << std::endl;
        pcout << "    pressure = " << n_p << std::endl;
        pcout << "    total    = " << n_u + n_p << std::endl;
    }

    pcout << "-----------------------------------------------" << std::endl;

    // Initialize the constraints (Changes made here)
    {
        constraints.clear();

        // Define boundary conditions
        std::map<types::boundary_id, const Function<dim> *> boundary_functions;
        Functions::ZeroFunction<dim> zero_function(dim + 1);
        boundary_functions[1] = &inlet_velocity;
        boundary_functions[3] = &zero_function;
        boundary_functions[4] = &zero_function;

        // Interpolate boundary values into the constraints
        VectorTools::interpolate_boundary_values(
            dof_handler,
            boundary_functions,
            constraints,
            ComponentMask({true, true, false, false}));

        // Close the constraints to prepare for use
        constraints.close();
    }

    // Initialize the linear system.
    {
        pcout << "Initializing the linear system" << std::endl;
        pcout << "  Initializing the sparsity pattern" << std::endl;

        Table<2, DoFTools::Coupling> coupling(dim + 1, dim + 1);
        for (unsigned int c = 0; c < dim + 1; ++c)
        {
            for (unsigned int d = 0; d < dim + 1; ++d)
            {
                coupling[c][d] = DoFTools::always; // Allow coupling for all components
            }
        }

        TrilinosWrappers::BlockSparsityPattern sparsity(block_owned_dofs, MPI_COMM_WORLD);
        DoFTools::make_sparsity_pattern(
            dof_handler, coupling, sparsity, constraints, /*keep_constrained_dofs=*/false);
        sparsity.compress();

        // Build sparsity pattern for the pressure mass matrix.
        for (unsigned int c = 0; c < dim + 1; ++c)
        {
            for (unsigned int d = 0; d < dim + 1; ++d)
            {
                if (c == dim && d == dim) // pressure-pressure term
                    coupling[c][d] = DoFTools::always;
                else // other combinations
                    coupling[c][d] = DoFTools::none;
            }
        }

        TrilinosWrappers::BlockSparsityPattern sparsity_pressure_mass(block_owned_dofs, MPI_COMM_WORLD);
        DoFTools::make_sparsity_pattern(
            dof_handler, coupling, sparsity_pressure_mass, constraints, /*keep_constrained_dofs=*/false);
        sparsity_pressure_mass.compress();

        pcout << "  Initializing the matrices" << std::endl;
        system_matrix.reinit(sparsity);
        pressure_mass.reinit(sparsity_pressure_mass);

        pcout << "  Initializing the system right-hand side" << std::endl;
        system_rhs.reinit(block_owned_dofs, MPI_COMM_WORLD);
        pcout << "  Initializing the solution vector" << std::endl;
        solution_owned.reinit(block_owned_dofs, MPI_COMM_WORLD);
        solution.reinit(block_owned_dofs, block_relevant_dofs, MPI_COMM_WORLD);
    }
}

void IncrementalStokes::assemble()
{
    const unsigned int dofs_per_cell = fe->dofs_per_cell;
    const unsigned int n_q = quadrature->size();
    const unsigned int n_q_face = quadrature_face->size();

    FEValues<dim> fe_values_restore(*fe, *quadrature,
                                    update_values | update_gradients | update_quadrature_points | update_JxW_values);

    FEFaceValues<dim> fe_face_values(*fe,
                                     *quadrature_face,
                                     update_values | update_normal_vectors |
                                         update_JxW_values);

    std::vector<types::global_dof_index> dof_indices(dofs_per_cell);
    FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double> local_rhs(dofs_per_cell);

    std::vector<Vector<double>> rhs_values(n_q, Vector<double>(dim + 1));

    std::vector<Tensor<1, dim>> previous_velocity_values(n_q);
    std::vector<Tensor<2, dim>> previous_velocity_gradients(n_q);
    std::vector<double> previous_pressure_values(n_q);

    std::vector<double> div_phi_u(dofs_per_cell);
    std::vector<Tensor<1, dim>> phi_u(dofs_per_cell);
    std::vector<Tensor<2, dim>> grad_phi_u(dofs_per_cell);
    std::vector<double> phi_p(dofs_per_cell);

    system_matrix = 0.0;
    system_rhs = 0.0;

    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        fe_values_restore.reinit(cell);

        local_matrix = 0.0;
        local_rhs = 0.0;

        fe_values_restore[u_k].get_function_values(solution_old, previous_velocity_values);
        fe_values_restore[u_k].get_function_gradients(solution_old, previous_velocity_gradients);
        fe_values_restore[p_k].get_function_values(solution_old, previous_pressure_values);

        for (unsigned int q = 0; q < n_q; q++)
        {
            for (unsigned int k = 0; k < dofs_per_cell; k++)
            {
                div_phi_u[k] = fe_values_restore[u_k].divergence(k, q);
                grad_phi_u[k] = fe_values_restore[u_k].gradient(k, q);
                phi_u[k] = fe_values_restore[u_k].value(k, q);
                phi_p[k] = fe_values_restore[p_k].value(k, q);
            }

            for (unsigned int i = 0; i < dofs_per_cell; i++)
            {

                for (unsigned int j = 0; j < dofs_per_cell; j++)
                {
                    local_matrix(i, j) += nu * scalar_product(grad_phi_u[i], grad_phi_u[j]) * fe_values_restore.JxW(q);                 // (I)
                    local_matrix(i, j) += phi_u[j] * transpose(previous_velocity_gradients[q]) * phi_u[i] * fe_values_restore.JxW(q);   // (II)
                    local_matrix(i, j) += previous_velocity_values[q] * transpose(grad_phi_u[j]) * phi_u[i] * fe_values_restore.JxW(q); // (III)
                    local_matrix(i, j) -= phi_p[j] * div_phi_u[i] * fe_values_restore.JxW(q);                                           // (IV)
                    local_matrix(i, j) -= phi_p[i] * div_phi_u[j] * fe_values_restore.JxW(q);                                           // (V)
                }

                local_rhs[i] += previous_velocity_values[q] * transpose(previous_velocity_gradients[q]) * phi_u[i] * fe_values_restore.JxW(q); // (VI)
            }
        }

        if (cell->at_boundary())
        {
            for (unsigned int f = 0; f < cell->n_faces(); ++f)
            {
                if (cell->face(f)->at_boundary() &&
                    cell->face(f)->boundary_id() == 2)
                {
                    fe_face_values.reinit(cell, f);

                    for (unsigned int q = 0; q < n_q_face; ++q)
                    {
                        for (unsigned int i = 0; i < dofs_per_cell; ++i)
                        {
                            local_rhs(i) += -p_out * scalar_product(fe_face_values.normal_vector(q), fe_face_values[u_k].value(i, q)) *
                                            fe_face_values.JxW(q);
                        }
                    }
                }
            }
        }

        cell->get_dof_indices(dof_indices);

        // Use constraints.distribute_local_to_global() instead of adding directly
        constraints.distribute_local_to_global(
            local_matrix,
            local_rhs,
            dof_indices,
            system_matrix,
            system_rhs);
    }

    pcout << "RHS Norm Value at iteration " << this->iter << " => " << system_rhs.l2_norm() << std::endl;

    this->iter++;

    // No need to call compress() here (Removed compress calls)
    // system_matrix.compress(VectorOperation::add);
    // system_rhs.compress(VectorOperation::add);
}

void IncrementalStokes::solve()
{
    for (unsigned int i = 0; i < maxIter; i++)
    {
        this->assemble();

        double update_norm = update_tol + 1;

        if (update_norm < update_tol)
            break;
        else
        {
            SolverControl solver_control(2000000, 1e-4);
            SolverGMRES<TrilinosWrappers::MPI::BlockVector> solver(solver_control);
            PreconditionIdentity preconditioner;
            // Apply constraints to the initial guess
            constraints.set_zero(solution_owned);
            solver.solve(system_matrix, solution_owned, system_rhs, preconditioner);
            // Distribute constraints to the solution
            constraints.distribute(solution_owned);

            pcout << "  " << solver_control.last_step() << " GMRES iterations" << std::endl;
            solution = solution_owned;
            new_res.reinit(solution);
            new_res = solution;
            new_res.sadd(1.0, -1.0, solution_old);

            if (mpi_rank == 0)
            {
                double sum = 0;
                for (size_t i = 0; i < new_res.size(); ++i)
                    sum += new_res(i) * new_res(i);
                update_norm = std::sqrt(sum);
                std::cout << "L2 norm of the update module = " << update_norm << std::endl;
            }

            solution_old = solution;
        }
    }
}

void IncrementalStokes::output()
{
    pcout << "===============================================" << std::endl;

    DataOut<dim> data_out;

    // Define correct interpretation for velocity and pressure
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
        data_component_interpretation(
            dim, DataComponentInterpretation::component_is_part_of_vector);
    data_component_interpretation.push_back(
        DataComponentInterpretation::component_is_scalar);

    // Make sure names match interpretation (e.g., 3D: velocity_x, velocity_y, velocity_z)
    std::vector<std::string> names;
    for (unsigned int i = 0; i < dim; ++i)
        names.push_back("velocity");
    names.push_back("pressure");

    // Add data vector
    data_out.add_data_vector(dof_handler, solution, names, data_component_interpretation);

    // Add partitioning information
    std::vector<unsigned int> partition_int(mesh.n_active_cells());
    GridTools::get_subdomain_association(mesh, partition_int);
    const Vector<double> partitioning(partition_int.begin(), partition_int.end());
    data_out.add_data_vector(partitioning, "partitioning");

    // Generate patches
    data_out.build_patches();

    std::string numProcessors = std::to_string(mpi_size);
    numProcessors += (mpi_size == 1) ? "_processor" : "_processors";

    // Set output file name
    const std::string output_file_name = "output-NavierStokes-" + numProcessors;

    std::string output_dir = get_output_directory();

    data_out.write_vtu_with_pvtu_record(output_dir,
                                        output_file_name,
                                        0,
                                        MPI_COMM_WORLD);
    pcout << "Output written to " << output_file_name << std::endl;
    pcout << "===============================================" << std::endl;
}

void IncrementalStokes::set_initial_conditions(TrilinosWrappers::MPI::BlockVector solution_stokes_)
{
    solution_old.reinit(solution_stokes_);
    this->solution_old = solution_stokes_;
}

//! NOT YET PARALLELIZED
void IncrementalStokes::compute_lift_drag()
{

    QGauss<dim - 1> face_quadrature_formula(3);
    const int n_q_points = face_quadrature_formula.size();

    const FEValuesExtractors::Vector velocities(0);
    const FEValuesExtractors::Scalar pressure(dim);

    std::vector<double> pressure_values(n_q_points);
    std::vector<Tensor<2, dim>> velocity_gradients(n_q_points);

    Tensor<1, dim> grad_u_tau;
    Tensor<1, dim> normal_vector;
    Tensor<1, dim> tangent_vector;

    Tensor<2, dim> fluid_stress;
    Tensor<2, dim> fluid_pressure;
    Tensor<1, dim> forces;

    FEFaceValues<dim> fe_face_values(*fe, face_quadrature_formula, update_values | update_quadrature_points | update_gradients | update_JxW_values | update_normal_vectors);

    double drag = 0;
    double lift = 0;

    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        if (cell->at_boundary())
        {

            for (unsigned int f = 0; f < cell->n_faces(); ++f)
            {
                fe_face_values.reinit(cell, f);
                std::vector<Point<dim>> q_points = fe_face_values.get_quadrature_points();

                if (cell->face(f)->boundary_id() == 4)
                {
                    fe_face_values[velocities].get_function_gradients(solution, velocity_gradients);
                    fe_face_values[pressure].get_function_values(solution, pressure_values);

                    for (int q = 0; q < n_q_points; q++)
                    {
                        normal_vector = -fe_face_values.normal_vector(q);

                        fluid_pressure.clear(); // Initialize to zero
                        fluid_pressure[0][0] = pressure_values[q];
                        fluid_pressure[1][1] = pressure_values[q];

                        fluid_stress = nu * velocity_gradients[q] - fluid_pressure;

                        forces = fluid_stress * normal_vector * fe_face_values.JxW(q);

                        drag += 500 * forces[0];
                        lift += 500 * forces[1];
                    }
                }
            }
        }
    }

    Point<dim> p1, p2;
    p1[0] = 0.15;
    p1[1] = 0.2;
    p2[0] = 0.25;
    p2[1] = 0.2;
    Vector<double> solution_values1(dim + 1);
    Vector<double> solution_values2(dim + 1);

    bool p1_available = true;
    bool p2_available = true;

    try
    {
        VectorTools::point_value(dof_handler, solution, p1, solution_values1);
    }
    catch (const dealii::VectorTools::ExcPointNotAvailableHere &)
    {
        p1_available = false;
    }

    try
    {
        VectorTools::point_value(dof_handler, solution, p2, solution_values2);
    }
    catch (const dealii::VectorTools::ExcPointNotAvailableHere &)
    {
        p2_available = false;
    }

    if (p1_available)
    {
        MPI_Send(&solution_values1[dim], 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }

    if (p2_available)
    {
        MPI_Send(&solution_values2[dim], 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }

    MPI_Send(&lift, 1, MPI_DOUBLE, 0, 2 * mpi_rank, MPI_COMM_WORLD);
    MPI_Send(&drag, 1, MPI_DOUBLE, 0, 2 * mpi_rank + 1, MPI_COMM_WORLD);

    if (mpi_rank == 0)
    {
        MPI_Status status;
        // Ricevi i dati da chiunque abbia inviato per p1
        MPI_Recv(&solution_values1[dim], 1, MPI_DOUBLE, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
        int source_p1 = status.MPI_SOURCE;

        // Ricevi i dati da chiunque abbia inviato per p2
        MPI_Recv(&solution_values2[dim], 1, MPI_DOUBLE, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
        int source_p2 = status.MPI_SOURCE;

        double p_diff = solution_values1(dim) - solution_values2(dim);

        for (int i = 1; i < mpi_size; i++)
        {
            double lift_i, drag_i;
            MPI_Recv(&lift_i, 1, MPI_DOUBLE, i, 2 * i, MPI_COMM_WORLD, &status);
            MPI_Recv(&drag_i, 1, MPI_DOUBLE, i, 2 * i + 1, MPI_COMM_WORLD, &status);
            lift += lift_i;
            drag += drag_i;
        }

        std::cout << "Pressure difference between points 1 and 2: " << p_diff << std::endl;

        namespace fs = std::filesystem;
        if (!fs::exists("./output/"))
        {
            fs::create_directory("../output");
        }

        std::string output_dir = get_output_directory();

        std::string filename = output_dir + "lift_drag_output.csv";

        FILE *pFile = fopen(filename.c_str(), "a");
        fprintf(pFile, "%f,  %f, %f\n", drag, lift, p_diff); // 500 = 2 / (U^2 * L)     // (L = 0.1, U = 2/3 * 0.3)
        fflush(pFile);
        fclose(pFile); // Close the file after writing
    }
}

std::string IncrementalStokes::get_output_directory()
{
    namespace fs = std::filesystem;

    if (!fs::exists("./outputs"))
    {
        fs::create_directory("./outputs");
    }

    if (!fs::exists("./outputs/steadyNavierStokes"))
    {
        fs::create_directory("./outputs/steadyNavierStokes");
    }

    std::string sub_dir_name = "outputs_reynolds_" + std::to_string(static_cast<int>(reynolds_number));
    fs::path sub_dir_path = "./outputs/steadyNavierStokes/" + sub_dir_name + "/";
    if (!fs::exists(sub_dir_path))
    {
        fs::create_directory(sub_dir_path);
    }
    fs::path sub_sub_dir_path = sub_dir_path.string() + "NavierStokes/";

    if (!fs::exists(sub_sub_dir_path))
    {
        fs::create_directory(sub_sub_dir_path);
    }

    return sub_sub_dir_path.string();
}
