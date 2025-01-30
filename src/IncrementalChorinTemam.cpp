#include "../include/IncrementalChorinTemam.hpp"

template <unsigned int dim>
void IncrementalChorinTemam<dim>::setup()
{

    std::cout << "Initializing the mesh" << std::endl;

    Triangulation<dim> mesh_serial;

    GridIn<dim> grid_in;
    grid_in.attach_triangulation(mesh_serial);

    std::ifstream grid_in_file(mesh_file_name);
    grid_in.read_msh(grid_in_file);

    GridTools::partition_triangulation(mpi_size, mesh_serial);
    const auto construction_data = TriangulationDescription::Utilities::
        create_description_from_triangulation(mesh_serial, MPI_COMM_WORLD);
    mesh.create_triangulation(construction_data);

    std::cout << "  Number of elements = " << mesh.n_global_active_cells()
              << std::endl;

    //-----------------------------
    // Velocity dofs
    //-----------------------------

    dof_handler_velocity.reinit(mesh);
    dof_handler_velocity.distribute_dofs(fe_velocity);
    locally_owned_velocity = dof_handler_velocity.locally_owned_dofs();
    locally_relevant_velocity = locally_owned_velocity;

    exact_velocity.set_time(0.0);

    constraints_velocity.clear();
    DoFTools::make_hanging_node_constraints(dof_handler_velocity, constraints_velocity);

    // Inlet velocity on boundary ID = 1:
    if constexpr (dim == 2)
    {
        VectorTools::interpolate_boundary_values(dof_handler_velocity,
                                                 /*boundary_id=*/1,
                                                 InletVelocity(H),
                                                 constraints_velocity);

        // Zero velocity (homogeneous Dirichlet) on boundary ID = 3:
        VectorTools::interpolate_boundary_values(dof_handler_velocity,
                                                 /*boundary_id=*/3,
                                                 Functions::ZeroFunction<dim>(dim),
                                                 constraints_velocity);

        // Zero velocity (homogeneous Dirichlet) on boundary ID = 4:
        VectorTools::interpolate_boundary_values(dof_handler_velocity,
                                                 /*boundary_id=*/4,
                                                 Functions::ZeroFunction<dim>(dim),
                                                 constraints_velocity);
    }
    else if constexpr (dim == 3)
    {
        VectorTools::interpolate_boundary_values(dof_handler_velocity,
                                                 /*boundary_id=*/0,
                                                 exact_velocity,
                                                 constraints_velocity);

        VectorTools::interpolate_boundary_values(dof_handler_velocity,
                                                 /*boundary_id=*/1,
                                                 exact_velocity,
                                                 constraints_velocity);

        // Zero velocity (homogeneous Dirichlet) on boundary ID = 3:
        VectorTools::interpolate_boundary_values(dof_handler_velocity,
                                                 /*boundary_id=*/2,
                                                 exact_velocity,
                                                 constraints_velocity);

        // Zero velocity (homogeneous Dirichlet) on boundary ID = 4:
        VectorTools::interpolate_boundary_values(dof_handler_velocity,
                                                 /*boundary_id=*/3,
                                                 exact_velocity,
                                                 constraints_velocity);

        VectorTools::interpolate_boundary_values(dof_handler_velocity,
                                                 /*boundary_id=*/5,
                                                 exact_velocity,
                                                 constraints_velocity);
    }
    constraints_velocity.close();

    //-----------------------------
    // Pressure dofs
    //-----------------------------
    dof_handler_pressure.reinit(mesh);
    dof_handler_pressure.distribute_dofs(fe_pressure);
    locally_owned_pressure = dof_handler_pressure.locally_owned_dofs();
    locally_relevant_pressure = locally_owned_pressure;

    constraints_pressure.clear();
    DoFTools::make_hanging_node_constraints(dof_handler_pressure, constraints_pressure);
    // Fix pressure=0 on boundary to remove nullspace
    if constexpr (dim == 2)
        VectorTools::interpolate_boundary_values(
            dof_handler_pressure,
            2,
            Functions::ZeroFunction<dim>(1), // pressure is scalar => "1" component
            constraints_pressure);
    else if constexpr (dim == 3)
        VectorTools::interpolate_boundary_values(
            dof_handler_pressure,
            4,
            Functions::ZeroFunction<dim>(1), // pressure is scalar => "1" component
            constraints_pressure);
    constraints_pressure.close();

    //-----------------------------
    // Sparsity patterns
    //-----------------------------

    {
        TrilinosWrappers::SparsityPattern dsp_v(locally_owned_velocity,
                                                MPI_COMM_WORLD);
        DoFTools::make_sparsity_pattern(dof_handler_velocity, dsp_v);
        dsp_v.compress();

        velocity_matrix.reinit(dsp_v);
        velocity_update_matrix.reinit(dsp_v);
    }
    {
        TrilinosWrappers::SparsityPattern dsp_p(locally_owned_pressure,
                                                MPI_COMM_WORLD);
        DoFTools::make_sparsity_pattern(dof_handler_pressure, dsp_p);
        dsp_p.compress();

        pressure_matrix.reinit(dsp_p);
    }

    //-----------------------------
    // Reinit all vectors
    //-----------------------------
    old_velocity.reinit(locally_owned_velocity, locally_relevant_velocity, MPI_COMM_WORLD);
    old_old_velocity.reinit(locally_owned_velocity, locally_relevant_velocity, MPI_COMM_WORLD);
    velocity_solution.reinit(locally_owned_velocity, locally_relevant_velocity, MPI_COMM_WORLD);
    update_velocity_solution.reinit(locally_owned_velocity, locally_relevant_velocity, MPI_COMM_WORLD);
    velocity_system_rhs.reinit(locally_owned_velocity, MPI_COMM_WORLD);
    velocity_update_rhs.reinit(locally_owned_velocity, MPI_COMM_WORLD);

    // old_pressure.reinit(locally_owned_pressure, locally_relevant_pressure, MPI_COMM_WORLD);
    deltap.reinit(locally_owned_pressure, locally_relevant_pressure, MPI_COMM_WORLD);
    pressure_solution.reinit(locally_owned_pressure, locally_relevant_pressure, MPI_COMM_WORLD);
    pressure_system_rhs.reinit(locally_owned_pressure, MPI_COMM_WORLD);

    std::cout << "DoFs: velocity=" << dof_handler_velocity.n_dofs()
              << ", pressure=" << dof_handler_pressure.n_dofs() << std::endl;
}

template <unsigned int dim>
void IncrementalChorinTemam<dim>::assemble_system_velocity()
{
    TimerOutput::Scope t(computing_timer, "assemble_velocity");

    velocity_matrix = 0;
    velocity_system_rhs = 0;

    const unsigned int quad_deg = std::max<unsigned int>(2u, fe_velocity.degree + 1u);
    QGaussSimplex<dim> quadrature_formula(quad_deg);
    QGaussSimplex<dim - 1> quadrature_boundary(fe_velocity.degree + 1);

    FEValues<dim> fe_values(fe_velocity, quadrature_formula,
                            update_values | update_gradients |
                                update_quadrature_points | update_JxW_values);

    FEValues<dim> fe_values_pressure(fe_pressure, quadrature_formula,
                                     update_gradients | update_quadrature_points);

    FEFaceValues<dim> fe_face_values(fe_velocity, quadrature_boundary,
                                     update_values | update_quadrature_points | update_JxW_values);

    FEValuesExtractors::Vector velocity(0);

    const unsigned int dofs_per_cell = fe_velocity.dofs_per_cell;
    const unsigned int n_q = quadrature_formula.size();

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double> cell_rhs(dofs_per_cell);
    std::vector<types::global_dof_index> local_indices(dofs_per_cell);

    std::vector<Tensor<1, dim>> old_val(n_q);
    std::vector<double> old_div(n_q);
    std::vector<Tensor<1, dim>> old_old_val(n_q);
    std::vector<double> old_old_div(n_q);
    std::vector<Tensor<1, dim>> pressure_grad(n_q);

    auto cell_v = dof_handler_velocity.begin_active();
    auto cell_p = dof_handler_pressure.begin_active();
    const auto end_v = dof_handler_velocity.end();

    Vector<double> neumann_loc(dim + 1);
    Tensor<1, dim> neumann_loc_tensor;

    for (; cell_v != end_v; ++cell_v, ++cell_p)
    {
        if (!cell_v->is_locally_owned())
            continue;

        fe_values.reinit(cell_v);
        fe_values_pressure.reinit(cell_p);

        cell_matrix = 0;
        cell_rhs = 0;

        const auto &vel_extract = fe_values[FEValuesExtractors::Vector(0)];
        vel_extract.get_function_values(old_velocity, old_val);
        vel_extract.get_function_divergences(old_velocity, old_div);
        vel_extract.get_function_values(old_old_velocity, old_old_val);
        vel_extract.get_function_divergences(old_old_velocity, old_old_div);

        fe_values_pressure.get_function_gradients(pressure_solution, pressure_grad);

        for (unsigned int q = 0; q < n_q; ++q)
        {
            const double JxW = fe_values.JxW(q);

            const Tensor<1, dim> &u_star = 2.0 * old_val[q] - old_old_val[q];
            const double u_star_div = 2.0 * old_div[q] - old_old_div[q];

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                const Tensor<1, dim> &phi_i = vel_extract.value(i, q);
                const Tensor<2, dim> &grad_phi_i = vel_extract.gradient(i, q);

                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                    const Tensor<1, dim> &phi_j = vel_extract.value(j, q);
                    const Tensor<2, dim> &grad_phi_j = vel_extract.gradient(j, q);

                    double lhs = 0.0;
                    // Mass
                    lhs += 3.0 * scalar_product(phi_j, phi_i);
                    // Viscous
                    lhs += 2.0 * deltat * nu * scalar_product(grad_phi_j, grad_phi_i);
                    // Convection
                    lhs += 2.0 * deltat * (scalar_product(grad_phi_j * u_star, phi_i) + 0.5 * scalar_product(phi_j * u_star_div, phi_i)); // dealii non condivide (o quasi)

                    cell_matrix(i, j) += lhs * JxW;
                }

                cell_rhs(i) += scalar_product(4.0 * old_val[q], phi_i) * JxW;
                cell_rhs(i) -= scalar_product(old_old_val[q], phi_i) * JxW;
                cell_rhs(i) -= 2.0 * deltat * scalar_product(pressure_grad[q], phi_i) * JxW;
            }
        }

        // functions.

        if (cell_v->at_boundary())
        {
            for (unsigned int face_number = 0; face_number < cell_v->n_faces();
                 ++face_number)
            {
                // If current face lies on the boundary, and its boundary ID (or
                // tag) is that of one of the Neumann boundaries, we assemble the
                // boundary integral.

                if (cell_v->face(face_number)->at_boundary() &&
                    (cell_v->face(face_number)->boundary_id() == 4))
                {
                    fe_face_values.reinit(cell_v, face_number);

                    for (unsigned int q = 0; q < quadrature_boundary.size(); ++q)
                    {
                        neumann_function.set_time(time);
                        neumann_function.vector_value(fe_face_values.quadrature_point(q),
                                                      neumann_loc);
                        for (unsigned int d = 0; d < dim; ++d)
                            neumann_loc_tensor[d] = neumann_loc[d];

                        for (unsigned int i = 0; i < dofs_per_cell; ++i)
                        {
                            cell_rhs(i) += 2.0 * deltat * scalar_product(neumann_loc_tensor,
                                                          fe_face_values[velocity].value(i, q)) *
                                           fe_face_values.JxW(q);
                        }

                        // neumann_function.set_time(time);
                        // for (unsigned int dimension = 0; dimension < dim; ++dimension)
                        // {
                        //     double n = neumann_function.value(fe_face_values.quadrature_point(q), dimension);

                        //     for (unsigned int i = 0; i < dofs_per_cell; ++i)
                        //     {
                        //         cell_rhs(i) += n * fe_face_values[velocity].value(i, q)[dimension] * fe_face_values.JxW(q);
                        //     }
                        // }
                    }
                }
            }
        }

        cell_v->get_dof_indices(local_indices);
        constraints_velocity.distribute_local_to_global(cell_matrix, cell_rhs,
                                                        local_indices,
                                                        velocity_matrix,
                                                        velocity_system_rhs);
    }

    velocity_matrix.compress(VectorOperation::add);
    velocity_system_rhs.compress(VectorOperation::add);
}

template <unsigned int dim>
void IncrementalChorinTemam<dim>::solve_velocity_system()
{
    TimerOutput::Scope t(computing_timer, "solve_velocity");

    TrilinosWrappers::MPI::Vector tmp(locally_owned_velocity, MPI_COMM_WORLD);

    SolverControl solver_control(1000000, 1e-12 * velocity_system_rhs.l2_norm());

    // Create and initialize preconditioner:
    TrilinosWrappers::PreconditionSSOR prec;
    prec.initialize(velocity_matrix);

    // Create GMRES solver *without* specifying any restart parameter:
    SolverGMRES<TrilinosWrappers::MPI::Vector> solver_gmres(solver_control);

    // Solve the linear system:
    solver_gmres.solve(velocity_matrix, tmp, velocity_system_rhs, prec);

    std::cout << "Velocity GMRES iterations: " << solver_control.last_step() << std::endl;

    // Distribute constraints (apply hanging-node constraints, Dirichlet BC, etc.):
    constraints_velocity.distribute(tmp);

    // Update the global velocity solution:
    velocity_solution = tmp;
}
template <unsigned int dim>
void IncrementalChorinTemam<dim>::assemble_system_pressure()
{

    TimerOutput::Scope t(computing_timer, "assemble_pressure");
    pressure_matrix = 0;
    pressure_system_rhs = 0;

    const unsigned int quad_deg = std::max<unsigned int>(2u, fe_pressure.degree + 1u);
    QGaussSimplex<dim> quad(quad_deg);
    QGaussSimplex<dim - 1> quad_face(quad_deg);

    FEValues<dim> fe_values_p(fe_pressure, quad,
                              update_values | update_gradients |
                                  update_quadrature_points | update_JxW_values);

    FEValues<dim> fe_values_v(fe_velocity, quad,
                              update_values | update_gradients | update_quadrature_points);

    FEFaceValues<dim> fe_face_values(fe_pressure, quad_face,
                                     update_values | update_quadrature_points | update_JxW_values);

    const unsigned int dofs_per_cell = fe_pressure.dofs_per_cell;
    const unsigned int n_q = quad.size();
    const unsigned int n_q_face = fe_face_values.n_quadrature_points;

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double> cell_rhs(dofs_per_cell);
    std::vector<types::global_dof_index> local_indices(dofs_per_cell);

    // We'll read velocity_solution for its divergence
    std::vector<double> div_u_star(n_q);

    auto cell_p = dof_handler_pressure.begin_active();
    auto cell_v = dof_handler_velocity.begin_active();
    const auto end_p = dof_handler_pressure.end();

    for (; cell_p != end_p; ++cell_p, ++cell_v)
    {
        if (!cell_p->is_locally_owned())
            continue;
        fe_values_p.reinit(cell_p);
        fe_values_v.reinit(cell_v);

        cell_matrix = 0;
        cell_rhs = 0;

        const auto &vel_extract = fe_values_v[FEValuesExtractors::Vector(0)];
        vel_extract.get_function_divergences(velocity_solution, div_u_star);

        for (unsigned int q = 0; q < n_q; ++q)
        {
            const double JxW = fe_values_p.JxW(q);
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                const double phi_i = fe_values_p.shape_value(i, q);
                const Tensor<1, dim> grad_i = fe_values_p.shape_grad(i, q);

                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                    const Tensor<1, dim> grad_j = fe_values_p.shape_grad(j, q);
                    cell_matrix(i, j) += scalar_product(grad_j, grad_i) * JxW;
                }
                // RHS = - 1/dt * div(u^*) * phi_i
                cell_rhs(i) -= 3.0 / (2.0 * deltat) * (div_u_star[q] * phi_i) * JxW;
            }
        } // We might also add boundary integrals if needed for outflow, etc.

        cell_p->get_dof_indices(local_indices);
        constraints_pressure.distribute_local_to_global(cell_matrix, cell_rhs,
                                                        local_indices,
                                                        pressure_matrix,
                                                        pressure_system_rhs);
    }

    pressure_matrix.compress(VectorOperation::add);
    pressure_system_rhs.compress(VectorOperation::add);
}

template <unsigned int dim>
void IncrementalChorinTemam<dim>::solve_pressure_system()
{
    TimerOutput::Scope t(computing_timer, "solve_pressure");

    TrilinosWrappers::MPI::Vector tmp(locally_owned_pressure, MPI_COMM_WORLD);
    SolverControl solver_control(2000000, 1e-12 * pressure_system_rhs.l2_norm());

    TrilinosWrappers::PreconditionIC prec;
    prec.initialize(pressure_matrix);

    SolverCG<TrilinosWrappers::MPI::Vector> solver_cg(solver_control);
    solver_cg.solve(pressure_matrix, tmp, pressure_system_rhs, prec);

    std::cout << "Pressure CG iterations: " << solver_control.last_step() << std::endl;

    constraints_pressure.distribute(tmp);
    deltap = tmp;
}

template <unsigned int dim>
void IncrementalChorinTemam<dim>::update_velocity()
{
    TimerOutput::Scope t(computing_timer, "assemble_update");

    velocity_update_matrix = 0;
    velocity_update_rhs = 0;

    const unsigned int quad_deg = std::max<unsigned int>(2u, fe_velocity.degree + 1u);
    QGaussSimplex<dim> quad(quad_deg);

    FEValues<dim> fe_values_vel(fe_velocity, quad,
                                update_values | update_gradients |
                                    update_quadrature_points | update_JxW_values);

    FEValues<dim> fe_values_p(fe_pressure, quad,
                              update_gradients | update_quadrature_points);

    const unsigned int dofs_per_cell = fe_velocity.dofs_per_cell;
    const unsigned int n_q = quad.size();

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double> cell_rhs(dofs_per_cell);
    std::vector<types::global_dof_index> local_indices(dofs_per_cell);

    std::vector<Tensor<1, dim>> u_tilde_vals(n_q);
    std::vector<Tensor<1, dim>> grad_delta_p(n_q);

    auto cell_v = dof_handler_velocity.begin_active();
    auto cell_p = dof_handler_pressure.begin_active();
    const auto end_v = dof_handler_velocity.end();

    for (; cell_v != end_v; ++cell_v, ++cell_p)
    {
        if (!cell_v->is_locally_owned())
            continue;

        fe_values_vel.reinit(cell_v);
        fe_values_p.reinit(cell_p);

        cell_matrix = 0;
        cell_rhs = 0;

        const auto &vel_extract = fe_values_vel[FEValuesExtractors::Vector(0)];
        vel_extract.get_function_values(velocity_solution, u_tilde_vals);

        fe_values_p.get_function_gradients(deltap, grad_delta_p);

        for (unsigned int q = 0; q < n_q; ++q)
        {
            const double JxW = fe_values_vel.JxW(q);

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                const Tensor<1, dim> &phi_i = vel_extract.value(i, q);

                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                    const Tensor<1, dim> &phi_j = vel_extract.value(j, q);
                    cell_matrix(i, j) += scalar_product(phi_i, phi_j) * JxW;
                }
                cell_rhs(i) += u_tilde_vals[q] * phi_i * JxW;
                cell_rhs(i) -= (2.0 / 3.0) * deltat * grad_delta_p[q] * phi_i * JxW;
            }
        }

        cell_v->get_dof_indices(local_indices);
        constraints_velocity.distribute_local_to_global(cell_matrix, cell_rhs,
                                                        local_indices,
                                                        velocity_update_matrix,
                                                        velocity_update_rhs);
    }
    velocity_update_matrix.compress(VectorOperation::add);
    velocity_update_rhs.compress(VectorOperation::add);
}

template <unsigned int dim>
void IncrementalChorinTemam<dim>::solve_update_velocity_system()
{

    TimerOutput::Scope t(computing_timer, "solve_update");

    TrilinosWrappers::MPI::Vector tmp(locally_owned_velocity, MPI_COMM_WORLD);
    SolverControl solver_control(2000, 1e-12 * velocity_update_rhs.l2_norm());

    // Jacobi or SSOR
    TrilinosWrappers::PreconditionJacobi::AdditionalData data;
    data.omega = 0.7;
    data.n_sweeps = 5;

    TrilinosWrappers::PreconditionJacobi prec;
    prec.initialize(velocity_update_matrix, data);

    SolverCG<TrilinosWrappers::MPI::Vector> solver_cg(solver_control);

    solver_cg.solve(velocity_update_matrix, tmp, velocity_update_rhs, prec);

    std::cout << "Velocity update CG iters: " << solver_control.last_step() << std::endl;

    constraints_velocity.distribute(tmp);
    update_velocity_solution = tmp;
}

template <unsigned int dim>
void IncrementalChorinTemam<dim>::output_results()
{
    DataOut<dim> data_out;

    std::vector<std::string> velocity_names(dim, "velocity");
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
        velocity_interpretation(dim,
                                DataComponentInterpretation::component_is_part_of_vector);

    data_out.attach_dof_handler(dof_handler_velocity);
    data_out.add_data_vector(update_velocity_solution,
                             velocity_names,
                             DataOut<dim>::type_dof_data,
                             velocity_interpretation);

    data_out.add_data_vector(dof_handler_pressure, pressure_solution, "pressure");
    data_out.add_data_vector(dof_handler_pressure, deltap, "deltap");
    data_out.build_patches();

    std::string output_dir = get_output_directory();

    // Usa un tempo formattato con padding (es. 000, 001, ..., 010)
    std::ostringstream fname;
    fname << output_dir.c_str();
    fname << "solution-0_" << std::setw(5) << std::setfill('0') << static_cast<int>(time * 1000) << ".vtk";

    std::ofstream out(fname.str());
    data_out.write_vtk(out);
}

template <unsigned int dim>
void IncrementalChorinTemam<dim>::run()
{
    setup();

    exact_velocity.set_time(0.0);
    exact_pressure.set_time(0.0);

    {
        TrilinosWrappers::MPI::Vector tmp(locally_owned_velocity, MPI_COMM_WORLD);
        VectorTools::interpolate(dof_handler_velocity,
                                 exact_velocity,
                                 tmp);
        constraints_velocity.distribute(tmp);
        old_velocity = tmp;
        old_old_velocity = tmp;
        velocity_solution = tmp;
    }

    {
        TrilinosWrappers::MPI::Vector tmp(locally_owned_pressure, MPI_COMM_WORLD);
        VectorTools::interpolate(dof_handler_pressure,
                                 exact_pressure,
                                 tmp);
        constraints_pressure.distribute(tmp);
        deltap = tmp;
        pressure_solution = tmp;
    }

    // Possibly output initial
    output_results();

    while (time < T - 0.5 * deltat)
    {
        time += deltat;
        ++time_step;

        time = deltat * time_step;
        std::cout << "\nTime step " << time_step << " at t=" << time << std::endl;

        exact_velocity.set_time(time);

        constraints_velocity.clear();

        VectorTools::interpolate_boundary_values(dof_handler_velocity,
                                                 /*boundary_id=*/0,
                                                 exact_velocity,
                                                 constraints_velocity);

        VectorTools::interpolate_boundary_values(dof_handler_velocity,
                                                 /*boundary_id=*/1,
                                                 exact_velocity,
                                                 constraints_velocity);

        // Zero velocity (homogeneous Dirichlet) on boundary ID = 3:
        VectorTools::interpolate_boundary_values(dof_handler_velocity,
                                                 /*boundary_id=*/2,
                                                 exact_velocity,
                                                 constraints_velocity);

        // Zero velocity (homogeneous Dirichlet) on boundary ID = 4:
        VectorTools::interpolate_boundary_values(dof_handler_velocity,
                                                 /*boundary_id=*/3,
                                                 exact_velocity,
                                                 constraints_velocity);

        VectorTools::interpolate_boundary_values(dof_handler_velocity,
                                                 /*boundary_id=*/5,
                                                 exact_velocity,
                                                 constraints_velocity);

        constraints_velocity.close();

        // 1) Intermediate velocity
        assemble_system_velocity();
        solve_velocity_system();

        // 2) Pressure
        assemble_system_pressure();
        solve_pressure_system();

        // 3) Velocity update
        update_velocity();
        solve_update_velocity_system();

        // Shift old velocities
        old_old_velocity = old_velocity;
        old_velocity = update_velocity_solution;

        // TrilinosWrappers::MPI::Vector tmp(locally_owned_velocity, MPI_COMM_WORLD);
        // VectorTools::interpolate(dof_handler_velocity,
        //                          exact_velocity,
        //                          tmp);
        // constraints_velocity.distribute(tmp);
        
        // update_velocity_solution = tmp;

        // pressure_solution = old_pressure + deltap;

        pressure_solution.add(deltap);

        output_results();
        // compute_lift_drag();

        // // Clear for next iteration
        // velocity_solution = 0;
        // pressure_solution = 0;
        // update_velocity_solution = 0;
        std::cout << compute_error(VectorTools::L2_norm) << std::endl;
        std::cout << compute_error(VectorTools::H1_norm) << std::endl;
    }
}

template <unsigned int dim>
void IncrementalChorinTemam<dim>::compute_lift_drag()
{
    // -------------------------------------------------
    // 1) Setup for face integration
    // -------------------------------------------------
    // Face quadrature for integration over the boundary
    const unsigned int face_quad_degree = 3;
    QGaussSimplex<dim - 1> face_quadrature_formula(face_quad_degree);

    // We will need separate FEFaceValues objects for velocity and pressure:
    FEFaceValues<dim> fe_face_values_velocity(
        fe_velocity,
        face_quadrature_formula,
        update_values | update_gradients | update_JxW_values | update_normal_vectors);

    FEFaceValues<dim> fe_face_values_pressure(
        fe_pressure,
        face_quadrature_formula,
        update_values | update_JxW_values);
    // ^ For pressure, we only need values (not gradients) if we do -p*I

    const unsigned int n_face_q_points = face_quadrature_formula.size();

    // We'll store velocity gradients and pressure values at the quadrature points
    std::vector<Tensor<2, dim>> velocity_gradients(n_face_q_points);
    std::vector<double> pressure_values(n_face_q_points);

    // Local partial sums of drag and lift on this MPI rank
    double local_drag = 0.0;
    double local_lift = 0.0;

    // -------------------------------------------------
    // 2) Loop over cells and faces to integrate traction
    // -------------------------------------------------
    // We assume mesh, dof_handler_velocity, and dof_handler_pressure
    // have the same Triangulation. So you can iterate them in parallel.
    auto cell_v = dof_handler_velocity.begin_active();
    auto cell_p = dof_handler_pressure.begin_active();
    const auto end_v = dof_handler_velocity.end();

    for (; cell_v != end_v; ++cell_v, ++cell_p)
    {
        if (!cell_v->is_locally_owned())
            continue;

        // Loop over faces:
        for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
        {
            // Check if this face is on the boundary of interest
            if (cell_v->face(f)->at_boundary() && (cell_v->face(f)->boundary_id() == 4))
            {
                // Reinit face-values for velocity and pressure
                fe_face_values_velocity.reinit(cell_v, f);
                fe_face_values_pressure.reinit(cell_p, f);

                // Extract velocity gradients at these face quadrature points
                fe_face_values_velocity[FEValuesExtractors::Vector(0)]
                    .get_function_gradients(velocity_solution, velocity_gradients);

                // Extract pressure values at these face quadrature points
                // (fe_pressure is scalar => use FEValuesExtractors::Scalar(0))
                fe_face_values_pressure[FEValuesExtractors::Scalar(0)]
                    .get_function_values(pressure_solution, pressure_values);

                // Now compute traction contributions
                for (unsigned int q = 0; q < n_face_q_points; ++q)
                {
                    const double p = pressure_values[q];
                    const Tensor<2, dim> grad_u = velocity_gradients[q];
                    const Tensor<1, dim> normal_vec = fe_face_values_velocity.normal_vector(q);
                    const double JxW = fe_face_values_velocity.JxW(q);

                    // If you want the symmetric gradient:
                    //   sym_grad_u = 0.5 * (grad_u + transpose(grad_u))
                    //   Here just do 2*nu*sym_grad_u = nu*(grad_u + grad_u^T)
                    //   so that fluid_stress = -p*I + 2*nu e(u).
                    // Or if you are consistent with your code in assemble, you might
                    // just do fluid_stress = -p I + nu*grad_u, etc.

                    Tensor<2, dim> sym_grad_u;
                    for (unsigned int d_i = 0; d_i < dim; ++d_i)
                        for (unsigned int d_j = 0; d_j < dim; ++d_j)
                            sym_grad_u[d_i][d_j] = 0.5 * (grad_u[d_i][d_j] + grad_u[d_j][d_i]);

                    // Construct the stress tensor = -p I + 2 nu e(u)
                    Tensor<2, dim> fluid_stress;
                    for (unsigned int d = 0; d < dim; ++d)
                        fluid_stress[d][d] = -p; // diagonal entries for -p*I

                    // Add 2 nu e(u)
                    for (unsigned int d_i = 0; d_i < dim; ++d_i)
                        for (unsigned int d_j = 0; d_j < dim; ++d_j)
                            fluid_stress[d_i][d_j] += 2.0 * nu * sym_grad_u[d_i][d_j];

                    // Traction = fluid_stress * normal
                    const Tensor<1, dim> traction = fluid_stress * normal_vec;

                    // Multiply by area element JxW
                    const Tensor<1, dim> force_contribution = traction * JxW;

                    // The x-component is "drag", the y-component is "lift" (2D assumption)
                    local_drag += force_contribution[0];
                    local_lift += force_contribution[1];
                } // end q-point loop
            }
        } // end face loop
    } // end cell loop

    // -------------------------------------------------
    // 3) Use MPI to sum up partial forces to root = rank 0
    // -------------------------------------------------
    double global_drag = 0.0, global_lift = 0.0;
    MPI_Allreduce(&local_drag, &global_drag, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&local_lift, &global_lift, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    // -------------------------------------------------
    // 4) Compute pressure difference between points p1 & p2
    // -------------------------------------------------
    // We only need the pressure dof_handler & solution here.
    // For each point, we'll see if it is "available" locally; if so,
    // we do point_value and send it to rank 0.  One simple approach:
    Point<dim> p1, p2;
    p1[0] = 0.15;
    p1[1] = 0.20;
    p2[0] = 0.25;
    p2[1] = 0.20;

    // Because pressure is scalar, we can store it in a double
    double local_p1 = 0.0, local_p2 = 0.0;
    bool have_p1 = false, have_p2 = false;

    try
    {
        local_p1 = VectorTools::point_value(dof_handler_pressure, pressure_solution, p1);
        have_p1 = true;
    }
    catch (...)
    {
        // This rank does not have p1
    }
    try
    {
        local_p2 = VectorTools::point_value(dof_handler_pressure, pressure_solution, p2);
        have_p2 = true;
    }
    catch (...)
    {
        // This rank does not have p2
    }

    // Send to root.  Alternatively, you can gather from all ranks, but
    // typically only one rank "owns" a given point in a distributed mesh.
    double p1_on_root = 0.0, p2_on_root = 0.0;
    int rank_p1 = -1;
    int rank_p2 = -1;

    // If we found p1, send it to root:
    if (have_p1)
    {
        MPI_Send(&local_p1, 1, MPI_DOUBLE, 0, 111, MPI_COMM_WORLD);
        MPI_Send(&mpi_rank, 1, MPI_INT, 0, 112, MPI_COMM_WORLD);
    }
    // If we found p2, send it to root:
    if (have_p2)
    {
        MPI_Send(&local_p2, 1, MPI_DOUBLE, 0, 221, MPI_COMM_WORLD);
        MPI_Send(&mpi_rank, 1, MPI_INT, 0, 222, MPI_COMM_WORLD);
    }

    // Only root receives
    if (mpi_rank == 0)
    {
        MPI_Status status;
        // We expect exactly one sender for p1, so:
        MPI_Recv(&p1_on_root, 1, MPI_DOUBLE, MPI_ANY_SOURCE, 111, MPI_COMM_WORLD, &status);
        MPI_Recv(&rank_p1, 1, MPI_INT, status.MPI_SOURCE, 112, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Same for p2:
        MPI_Recv(&p2_on_root, 1, MPI_DOUBLE, MPI_ANY_SOURCE, 221, MPI_COMM_WORLD, &status);
        MPI_Recv(&rank_p2, 1, MPI_INT, status.MPI_SOURCE, 222, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        const double p_diff = p1_on_root - p2_on_root;

        // -------------------------------------------------
        // 5) Print or write to a file
        // -------------------------------------------------
        // std::cout << "Time = " << time << "  Drag = " << global_drag
        //           << "  Lift = " << global_lift
        //           << "  p_diff(p1-p2) = " << p_diff << std::endl;

        // Append to a CSV file:

        std::string output_dir = get_output_directory();

        std::string filename = output_dir + "lift_drag_output.csv";

        std::ofstream out_file(filename.c_str(), std::ios::app);
        out_file << time << ","
                 << global_drag << ","
                 << global_lift << ","
                 << p_diff << "\n";
        out_file.close();
    }
}

template <unsigned int dim>
std::string IncrementalChorinTemam<dim>::get_output_directory()
{

    namespace fs = std::filesystem;

    if (!fs::exists("./outputs"))
        fs::create_directory("./outputs");
    if constexpr (dim == 2)
    {
        if (!fs::exists("./outputs/IncrementalChorinTemam2D"))
        {
            fs::create_directory("./outputs/IncrementalChorinTemam2D");
        }
    }
    else if constexpr (dim == 3)
    {
        if (!fs::exists("./outputs/IncrementalChorinTemam3D"))
            fs::create_directory("./outputs/IncrementalChorinTemam3D");
    }

    fs::path sub_dir_path = "";

    std::string sub_dir_name = "outputs_reynolds_" + std::to_string(static_cast<int>(reynolds_number));
    if constexpr (dim == 2)
        sub_dir_path = "./outputs/IncrementalChorinTemam2D/" + sub_dir_name + "/";
    else if constexpr (dim == 3)
        sub_dir_path = "./outputs/IncrementalChorinTemam3D/" + sub_dir_name + "/";

    if (!fs::exists(sub_dir_path))
    {
        fs::create_directory(sub_dir_path);
    }

    return sub_dir_path.string();
}

template <unsigned int dim>
double IncrementalChorinTemam<dim>::compute_error(const VectorTools::NormType &norm_type)
{
    // Define the quadrature for error computation
    const unsigned int quad_deg = fe_velocity.degree + 2; // Degree of quadrature rule
    QGaussSimplex<dim> quadrature_formula(quad_deg);

    // Define FEValues object for extracting velocity components
    FEValues<dim> fe_values(fe_velocity, quadrature_formula,
                            update_values | update_quadrature_points | update_JxW_values);

    // Exact solution setup
    exact_velocity.set_time(time);

    // Containers to hold exact and computed values
    std::vector<Vector<double>> exact_velocity_values(quadrature_formula.size(), Vector<double>(dim));
    std::vector<Tensor<1, dim>> computed_velocity_values(quadrature_formula.size());

    double local_error = 0.0;

    for (const auto &cell : dof_handler_velocity.active_cell_iterators())
    {
        if (!cell->is_locally_owned())
            continue;

        fe_values.reinit(cell);

        // Extract the computed velocity at quadrature points
        fe_values[FEValuesExtractors::Vector(0)].get_function_values(update_velocity_solution, computed_velocity_values);

        // Compute the exact velocity at quadrature points
        exact_velocity.vector_value_list(fe_values.get_quadrature_points(), exact_velocity_values);

        // Convert exact_velocity_values to Tensor<1, dim>
        std::vector<Tensor<1, dim>> exact_velocity_tensors(quadrature_formula.size());
        for (unsigned int q = 0; q < quadrature_formula.size(); ++q)
        {
            for (unsigned int i = 0; i < dim; ++i)
            {
                exact_velocity_tensors[q][i] = exact_velocity_values[q][i];
            }
        }

        // Integrate the error over the cell
        for (unsigned int q = 0; q < quadrature_formula.size(); ++q)
        {
            const double JxW = fe_values.JxW(q);

            // Compute the L2-norm of the error at this quadrature point
            Tensor<1, dim> error = computed_velocity_values[q] - exact_velocity_tensors[q];
            local_error += error.norm_square() * JxW;
        }
    }

    // Reduce the error across all processes
    double global_error = 0.0;
    MPI_Allreduce(&local_error, &global_error, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    // Return the square root of the global error for the L2 norm
    return std::sqrt(global_error);
}

template class IncrementalChorinTemam<2>;
template class IncrementalChorinTemam<3>;
