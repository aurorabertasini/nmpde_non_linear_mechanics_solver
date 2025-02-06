#include "../include/MonolithicNavierStokes.hpp"
#include "../include/preconditioners.hpp"

template <unsigned int dim>
void MonolithicNavierStokes<dim>::setup()
{
    // Create the mesh.
    {
        pcout << "Initializing the mesh" << std::endl;

        Triangulation<dim> mesh_serial;

        GridIn<dim> grid_in;
        grid_in.attach_triangulation(mesh_serial);

        std::ifstream grid_in_file(mesh_file_name);
        grid_in.read_msh(grid_in_file);

        GridTools::partition_triangulation(mpi_size, mesh_serial);
        const auto construction_data = TriangulationDescription::Utilities::
            create_description_from_triangulation(mesh_serial, MPI_COMM_WORLD);
        mesh.create_triangulation(construction_data);

        pcout << "  Number of elements = " << mesh.n_global_active_cells()
              << std::endl;
    }

    pcout << "-----------------------------------------------" << std::endl;

    // Initialize the finite element space.
    {
        pcout << "Initializing the finite element space" << std::endl;

        const FE_SimplexP<dim> fe_scalar_velocity(degree_velocity);
        const FE_SimplexP<dim> fe_scalar_pressure(degree_pressure);

        fe = std::make_unique<FESystem<dim>>(
            fe_scalar_velocity,
            dim,
            fe_scalar_pressure,
            1);

        pcout << "  Velocity degree:           = " << fe_scalar_velocity.degree
              << std::endl;
        pcout << "  Pressure degree:           = " << fe_scalar_pressure.degree
              << std::endl;
        pcout << "  DoFs per cell              = " << fe->dofs_per_cell
              << std::endl;

        quadrature = std::make_unique<QGaussSimplex<dim>>(fe->degree + 1);

        pcout << "  Quadrature points per cell = " << quadrature->size()
              << std::endl;

        quadrature_face = std::make_unique<QGaussSimplex<dim - 1>>(fe->degree + 1);

        pcout << "  Quadrature points per face = " << quadrature_face->size()
              << std::endl;
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

    // Initialize the linear system.
    {
        pcout << "Initializing the linear system" << std::endl;
        pcout << "  Initializing the sparsity pattern" << std::endl;

        Table<2, DoFTools::Coupling> coupling(dim + 1, dim + 1);
        for (unsigned int c = 0; c < dim + 1; ++c)
        {
            for (unsigned int d = 0; d < dim + 1; ++d)
            {
                if (c == dim && d == dim) // pressure-pressure term
                    coupling[c][d] = DoFTools::none;
                else // other combinations
                    coupling[c][d] = DoFTools::always;
            }
        }

        TrilinosWrappers::BlockSparsityPattern sparsity(block_owned_dofs,
                                                        MPI_COMM_WORLD);
        DoFTools::make_sparsity_pattern(dof_handler, coupling, sparsity);
        sparsity.compress();

        // Do the same for the velocity mass term.
        for (unsigned int c = 0; c < dim + 1; ++c)
        {
            for (unsigned int d = 0; d < dim + 1; ++d)
            {
                if (c == dim || d == dim) // terms with pressure
                    coupling[c][d] = DoFTools::none;
                else // terms with no pressure
                    coupling[c][d] = DoFTools::always;
            }
        }

        TrilinosWrappers::BlockSparsityPattern velocity_mass_sparsity(
            block_owned_dofs, MPI_COMM_WORLD);
        DoFTools::make_sparsity_pattern(dof_handler, coupling,
                                        velocity_mass_sparsity);
        velocity_mass_sparsity.compress();

        // Do the same for the pressure mass term.
        TrilinosWrappers::BlockSparsityPattern pressure_mass_sparsity(
            block_owned_dofs, MPI_COMM_WORLD);
        if (true)
        {
            for (unsigned int c = 0; c < dim + 1; ++c)
            {
                for (unsigned int d = 0; d < dim + 1; ++d)
                {
                    if (c == dim && d == dim) // terms with only pressure
                        coupling[c][d] = DoFTools::always;
                    else // terms with velocity
                        coupling[c][d] = DoFTools::none;
                }
            }

            DoFTools::make_sparsity_pattern(dof_handler, coupling,
                                            pressure_mass_sparsity);
            pressure_mass_sparsity.compress();
        }

        pcout << "  Initializing the matrices" << std::endl;

        system_matrix.reinit(sparsity);
        velocity_mass.reinit(velocity_mass_sparsity);
        pressure_mass.reinit(pressure_mass_sparsity);

        pcout << "  Initializing the system right-hand side" << std::endl;
        system_rhs.reinit(block_owned_dofs, MPI_COMM_WORLD);

        pcout << "  Initializing the solution vector" << std::endl;
        solution_owned.reinit(block_owned_dofs, MPI_COMM_WORLD);
        solution.reinit(block_owned_dofs, block_relevant_dofs, MPI_COMM_WORLD);
        // solution.update_ghost_values();
    }
}

template <unsigned int dim>
void MonolithicNavierStokes<dim>::assemble_base_matrix()
{
    const unsigned int dofs_per_cell = fe->dofs_per_cell;
    const unsigned int n_q = quadrature->size();

    FEValues<dim> fe_values(*fe,
                            *quadrature,
                            update_values | update_gradients |
                                update_quadrature_points | update_JxW_values);

    FullMatrix<double> cell_system_matrix(dofs_per_cell, dofs_per_cell);

    std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

    std::vector<Tensor<1, dim>> previous_velocity_values(n_q);
    std::vector<double> previous_velocity_divergence(n_q);
    // std::cout << "system_matrix initialized to 0.0" << std::endl;
    FullMatrix<double> velocity_mass_cell_matrix(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> pressure_mass_cell_matrix(dofs_per_cell, dofs_per_cell);
    // std::cout << "pressure_mass_cell_matrix initialized to 0.0" << std::endl;
    velocity_mass = 0.0;
    pressure_mass = 0.0;
    system_matrix = 0.0;
    // std::cout << "pressure_mass initialized to 0.0" << std::endl;

    // std::cout << "Starting assembly of base matrix" << std::endl;

    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        if (!cell->is_locally_owned())
            continue;

        fe_values.reinit(cell);
        // std::cout << "fe_values reinitialized for cell" << std::endl;

        std::vector<double> div_phi_u(dofs_per_cell);
        std::vector<Tensor<1, dim>> phi_u(dofs_per_cell);
        std::vector<Tensor<2, dim>> grad_phi_u(dofs_per_cell);
        std::vector<double> phi_p(dofs_per_cell);

        pressure_mass_cell_matrix = 0.0;
        velocity_mass_cell_matrix = 0.0;
        cell_system_matrix = 0.0;

        fe_values[velocity].get_function_values(solution, previous_velocity_values);
        fe_values[velocity].get_function_divergences(solution, previous_velocity_divergence);

        for (unsigned int q = 0; q < n_q; ++q)
        {
            for (unsigned int k = 0; k < dofs_per_cell; k++)
            {
                div_phi_u[k] = fe_values[velocity].divergence(k, q);
                grad_phi_u[k] = fe_values[velocity].gradient(k, q);
                phi_u[k] = fe_values[velocity].value(k, q);
                phi_p[k] = fe_values[pressure].value(k, q);
            }
            // std::cout << "Values at quadrature point " << q << " obtained" << std::endl;

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                    // mass component
                    cell_system_matrix(i, j) += scalar_product(phi_u[i], phi_u[j]) / deltat * fe_values.JxW(q);
                    velocity_mass_cell_matrix(i, j) += scalar_product(phi_u[i], phi_u[j]) / deltat * fe_values.JxW(q);
                    // stiffness component
                    cell_system_matrix(i, j) += nu * scalar_product(grad_phi_u[i], grad_phi_u[j]) * fe_values.JxW(q);
                    // Pressure term in the momentum equation
                    cell_system_matrix(i, j) -= phi_p[i] * div_phi_u[j] * fe_values.JxW(q);
                    // Pressure term in the continuity equation
                    cell_system_matrix(i, j) -= phi_p[j] * div_phi_u[i] * fe_values.JxW(q);
                    // Non-linear term
                    cell_system_matrix(i, j) += scalar_product(grad_phi_u[j] * previous_velocity_values[q], phi_u[i]) * fe_values.JxW(q);
                    // skew-symmetric term
                    cell_system_matrix(i, j) += 0.5 * previous_velocity_divergence[q] * scalar_product(phi_u[i], phi_u[j]) * fe_values.JxW(q);

                    pressure_mass_cell_matrix(i, j) += fe_values[pressure].value(j, q) *
                                                       fe_values[pressure].value(i, q) /
                                                       nu * fe_values.JxW(q);
                }
            }
            // std::cout << "cell_system_matrix and pressure_mass_cell_matrix updated for quadrature point " << q << std::endl;
        }
        cell->get_dof_indices(dof_indices);
        // std::cout << "dof_indices obtained for cell" << std::endl;

        system_matrix.add(dof_indices, cell_system_matrix);
        velocity_mass.add(dof_indices, velocity_mass_cell_matrix);
        pressure_mass.add(dof_indices, pressure_mass_cell_matrix);
        // std::cout << "system_matrix and pressure_mass updated for cell" << std::endl;
    }
    system_matrix.compress(VectorOperation::add);
    pressure_mass.compress(VectorOperation::add);
    velocity_mass.compress(VectorOperation::add);
    // std::cout << "system_matrix and pressure_mass compressed" << std::endl;

    // std::cout << "Finished assembly of base matrix" << std::endl;
}

template <unsigned int dim>
void MonolithicNavierStokes<dim>::assemble_rhs(const double &time)
{
    const unsigned int dofs_per_cell = fe->dofs_per_cell;
    const unsigned int n_q = quadrature->size();
    const unsigned int n_q_face = quadrature_face->size();

    FEValues<dim> fe_values(*fe,
                            *quadrature,
                            update_values | update_gradients | update_quadrature_points |
                                update_JxW_values);

    FEFaceValues<dim> fe_face_values(*fe,
                                     *quadrature_face,
                                     update_values |
                                         update_normal_vectors | update_quadrature_points |
                                         update_JxW_values);

    Vector<double> cell_rhs(dofs_per_cell);

    std::vector<Tensor<1, dim>> previous_velocity_values(n_q);

    std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

    system_rhs = 0.0;

    Vector<double> f_neumann_loc(dim + 1);

    Tensor<1, dim> f_neumann_tensor;

    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        if (!cell->is_locally_owned())
            continue;
        fe_values.reinit(cell);

        fe_values[velocity].get_function_values(solution, previous_velocity_values);

        cell_rhs = 0.0;
        for (unsigned int q = 0; q < n_q; ++q)
        {
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                cell_rhs(i) += scalar_product(previous_velocity_values[q],
                                              fe_values[velocity].value(i, q)) *
                               fe_values.JxW(q) / deltat;
            }
        }
        if (cell->at_boundary())
        {
            exact_solution3D.set_time(time);
            for (unsigned int f = 0; f < cell->n_faces(); ++f)
            {
                if (cell->face(f)->at_boundary() &&
                    cell->face(f)->boundary_id() == 4)
                {
                    fe_face_values.reinit(cell, f);

                    for (unsigned int q = 0; q < n_q_face; ++q)
                    {
                        exact_solution3D.neumann_condition.vector_value(
                            fe_face_values.quadrature_point(q), f_neumann_loc);


                        for (unsigned int d = 0; d < dim; ++d)
                            f_neumann_tensor[d] = f_neumann_loc(d);


                        for (unsigned int i = 0; i < dofs_per_cell; ++i)
                        {
                            cell_rhs(i) +=
                                scalar_product(f_neumann_tensor,
                                               fe_face_values[velocity].value(i, q)) *
                                fe_face_values.JxW(q);
                        }
                    }
                }
            }
        }
        cell->get_dof_indices(dof_indices);
        system_rhs.add(dof_indices, cell_rhs);
    }

    system_rhs.compress(VectorOperation::add);

    {
        std::map<types::global_dof_index, double> boundary_values;

        std::map<types::boundary_id, const Function<dim> *> boundary_functions;
        // IMPORTANT: if i use exact_solutio(t);
        //  exact_solution.set_time(time);
        //  boundary_functions[0] = &exact_solution;

        // if i dont know the exact solution
        exact_solution3D.set_time(time);
        boundary_functions[0] = &exact_solution3D;
        boundary_functions[1] = &exact_solution3D;
        boundary_functions[2] = &exact_solution3D;
        boundary_functions[3] = &exact_solution3D;
        boundary_functions[5] = &exact_solution3D;

        VectorTools::interpolate_boundary_values(dof_handler,
                                                 boundary_functions,
                                                 boundary_values,
                                                 ComponentMask({true, true, true , false}));

        MatrixTools::apply_boundary_values(
            boundary_values, system_matrix, solution_owned, system_rhs, false);
    }
}

template <unsigned int dim>
void MonolithicNavierStokes<dim>::solve_time_step()
{
    // Choose the preconditioner type:
    // 1 = SIMPLE, 2 = ASIMPLE, 3 = YOSIDA, 4 = AYOSIDA.
    static constexpr int precond_type = 1; // Set this value as needed

    // Local parameters for inner solvers and preconditioner initialization.
    static constexpr double alpha = 1;                   // Damping parameter for SIMPLE-like preconditioners
    static constexpr unsigned int maxiter_inner = 10000; // Maximum iterations for inner solvers
    static constexpr double tol_inner = 1e-5;            // Tolerance for inner solvers
    static constexpr bool use_ilu = false;               // Flag: true for ILU, false for AMG

    // Set up the outer GMRES solver.
    SolverControl solver_control(10000, 1e-7);
    SolverGMRES<TrilinosWrappers::MPI::BlockVector> solver(solver_control);

    std::shared_ptr<BlockPrecondition> block_precondition;

    // Select and initialize the preconditioner based on precond_type.
    switch (precond_type)
    {
    case 1:
    {
        auto simple_precondition = std::make_shared<PreconditionSIMPLE>();
        simple_precondition->initialize(
            system_matrix.block(0, 0),
            system_matrix.block(1, 0),
            system_matrix.block(0, 1),
            solution_owned,
            alpha,
            maxiter_inner,
            tol_inner,
            use_ilu);
        block_precondition = simple_precondition;
        break;
    }
    case 2:
    {
        auto asimple_precondition = std::make_shared<PreconditionaSIMPLE>();
        asimple_precondition->initialize(
            system_matrix.block(0, 0),
            system_matrix.block(1, 0),
            system_matrix.block(0, 1),
            solution_owned,
            alpha,
            maxiter_inner,
            tol_inner,
            use_ilu);
        block_precondition = asimple_precondition;
        break;
    }
    case 3:
    {
        auto yosida_precondition = std::make_shared<PreconditionYosida>();
        yosida_precondition->initialize(
            system_matrix.block(0, 0),
            system_matrix.block(1, 0),
            system_matrix.block(0, 1),
            velocity_mass.block(0, 0),
            solution_owned,
            maxiter_inner,
            tol_inner,
            use_ilu);
        block_precondition = yosida_precondition;
        break;
    }
    default:
        Assert(false, ExcNotImplemented());
    }

    solver.solve(system_matrix,
                 solution_owned,
                 system_rhs,
                 *block_precondition);

    pcout << "  " << solver_control.last_step() << " GMRES iterations" << std::endl;

    solution = solution_owned;

    solution.update_ghost_values();
}

template <unsigned int dim>
void MonolithicNavierStokes<dim>::solve()
{
    pcout << "===============================================" << std::endl;

    time = 0.0;

    // Apply the initial condition.
    {
        pcout << "Applying the initial condition" << std::endl;

        exact_solution3D.set_time(time);
        VectorTools::interpolate(dof_handler, exact_solution3D, solution_owned, ComponentMask({true, true, true, false}));
        solution = solution_owned;

        pcout << "-----------------------------------------------" << std::endl;
    }

    unsigned int time_step = 0;

    while (time < T - 0.5 * deltat)
    {
        time += deltat;
        ++time_step;

        pcout << "n = " << std::setw(3) << time_step << ", t = " << std::setw(5)
              << time << ":" << std::flush;

        exact_solution3D.set_time(time);

        assemble_base_matrix();
        assemble_rhs(time);
        solve_time_step();

        l2_H1_error += compute_error(VectorTools::H1_norm, true) * compute_error(VectorTools::H1_norm, true);

        output(time_step);
    }
}

template <unsigned int dim>
void MonolithicNavierStokes<dim>::output(const unsigned int &time_step)
{
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
    const Vector<float> partitioning(partition_int.begin(), partition_int.end());
    data_out.add_data_vector(partitioning, "partitioning");

    // Generate patches
    // solution.update_ghost_values();

    data_out.build_patches();

    // std::string numProcessors = std::to_string(mpi_size);
    // numProcessors += (mpi_size == 1) ? "_processor" : "_processors";

    std::string output_dir = get_output_directory();

    // Set output file name

    // Write output to VTU/PVTU
    // data_out.write_vtu_with_pvtu_record("./output-IncrementalStokes",
    //                                     time_step,
    //                                     MPI_COMM_WORLD,3);

    data_out.write_vtu_with_pvtu_record(
        output_dir, "output_", time_step, MPI_COMM_WORLD, 3);
}

template <unsigned int dim>
void MonolithicNavierStokes<dim>::update_buondary_conditions()
{
}

template <unsigned int dim>
std::string MonolithicNavierStokes<dim>::get_output_directory()
{
    namespace fs = std::filesystem;

    if (!fs::exists("./outputs"))
    {
        fs::create_directory("./outputs");
    }
    if constexpr (dim == 2)
    {
        if (!fs::exists("./outputs/monolithicNavierStokes2D"))
        {
            fs::create_directory("./outputs/monolithicNavierStokes2D");
        }
    }
    else if constexpr (dim == 3)
    {
        if (!fs::exists("./outputs/monolithicNavierStokes3D"))
        {
            fs::create_directory("./outputs/monolithicNavierStokes3D");
        }
    }

    std::string sub_dir_name = "outputs_";
    fs::path sub_dir_path = "";
    if constexpr (dim == 2)
        sub_dir_path = "./outputs/monolithicNavierStokes2D/" + sub_dir_name + "/";
    if constexpr (dim == 3)
        sub_dir_path = "./outputs/monolithicNavierStokes3D/" + sub_dir_name + "/";
    if (!fs::exists(sub_dir_path))
    {
        fs::create_directory(sub_dir_path);
    }

    return sub_dir_path.string();
}

template <unsigned int dim>
double MonolithicNavierStokes<dim>::compute_error(const VectorTools::NormType &norm_type, bool velocity)
{
    FE_SimplexP<dim> fe_mapping(1);
    MappingFE mapping(fe_mapping);

    // First we compute the norm on each element, and store it in a vector.
    Vector<double> error_per_cell(mesh.n_active_cells());

    if (velocity)
    {
        // The error is an integral, and we approximate that integral using a
        // quadrature formula. To make sure we are accurate enough, we use a
        // quadrature formula with one node more than what we used in
        // assembly.
        QGaussSimplex<dim> quadrature_error(degree_velocity + 2);

        ComponentSelectFunction<dim> mask(std::make_pair(0, dim), dim + 1);

        VectorTools::integrate_difference(
            mapping, dof_handler, solution, exact_solution3D.exact_velocity,
            error_per_cell, quadrature_error, norm_type, &mask);
    }
    else
    {
        // Do the same for the pressure.
        QGaussSimplex<dim> quadrature_error(this->degree_pressure + 2);
        ComponentSelectFunction<dim> mask(dim, dim + 1);
        VectorTools::integrate_difference(mapping, this->dof_handler, this->solution,
                                          exact_solution3D.exact_pressure, error_per_cell,
                                          quadrature_error, norm_type, &mask);
    }

    // Then, we add out all the cells.
    const double error =
        VectorTools::compute_global_error(mesh, error_per_cell, norm_type);

    return error;
}

template <unsigned int dim>
double MonolithicNavierStokes<dim>::get_l2_H1_error()
{
    return sqrt(deltat * l2_H1_error);
}


template class MonolithicNavierStokes<2>;
template class MonolithicNavierStokes<3>;