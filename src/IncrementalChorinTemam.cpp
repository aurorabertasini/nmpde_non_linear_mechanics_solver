#include "../include/IncrementalChorinTemam.hpp"

template <unsigned int dim>
void IncrementalChorinTemam<dim>::setup()
{
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

    exact_velocity3D.set_time(0.0);

    constraints_velocity.clear();
    DoFTools::make_hanging_node_constraints(dof_handler_velocity, constraints_velocity);

    // Inlet velocity on boundary ID = 1:
    if constexpr (dim == 2)
    {
        for (int i = 0; i < 4; i++)
            VectorTools::interpolate_boundary_values(dof_handler_velocity,
                                                     /*boundary_id=*/i,
                                                     Functions::ZeroFunction<dim>(dim),
                                                     constraints_velocity);
    }
    else if constexpr (dim == 3)
    {
        VectorTools::interpolate_boundary_values(dof_handler_velocity,
                                                 /*boundary_id=*/0,
                                                 exact_velocity3D,
                                                 constraints_velocity);

        VectorTools::interpolate_boundary_values(dof_handler_velocity,
                                                 /*boundary_id=*/1,
                                                 exact_velocity3D,
                                                 constraints_velocity);

        // Zero velocity (homogeneous Dirichlet) on boundary ID = 3:
        VectorTools::interpolate_boundary_values(dof_handler_velocity,
                                                 /*boundary_id=*/2,
                                                 exact_velocity3D,
                                                 constraints_velocity);

        // Zero velocity (homogeneous Dirichlet) on boundary ID = 4:
        VectorTools::interpolate_boundary_values(dof_handler_velocity,
                                                 /*boundary_id=*/3,
                                                 exact_velocity3D,
                                                 constraints_velocity);

        VectorTools::interpolate_boundary_values(dof_handler_velocity,
                                                 /*boundary_id=*/5,
                                                 exact_velocity3D,
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
    if constexpr (dim == 3)
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
            Tensor<1, dim> forcing_term_new_tensor;
            if constexpr (dim == 2)
            {
                Vector<double> f_new_loc(dim);
                forcing_term2D.set_time(time);
                forcing_term2D.vector_value(fe_values.quadrature_point(q),
                                            f_new_loc);
                for (unsigned int d = 0; d < dim; ++d)
                    forcing_term_new_tensor[d] = f_new_loc[d];
            }

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
                    lhs += 2.0 * deltat * (scalar_product(grad_phi_j * u_star, phi_i) ); // dealii non condivide (o quasi)

                    cell_matrix(i, j) += lhs * JxW;
                }

                cell_rhs(i) += scalar_product(4.0 * old_val[q], phi_i) * JxW;
                cell_rhs(i) -= scalar_product(old_old_val[q], phi_i) * JxW;
                cell_rhs(i) -= 2.0 * deltat * scalar_product(pressure_grad[q], phi_i) * JxW;
                if constexpr (dim == 2)
                {
                    cell_rhs(i) += 2.0 * deltat * scalar_product(forcing_term_new_tensor, phi_i) * JxW;
                }
            }
        }

        // functions.

        if constexpr (dim == 3) // neumann buondary on velocity for the 3d benchmark

        {
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
                            neumann_function3D.set_time(time);
                            neumann_function3D.vector_value(fe_face_values.quadrature_point(q),
                                                            neumann_loc);
                            for (unsigned int d = 0; d < dim; ++d)
                                neumann_loc_tensor[d] = neumann_loc[d];

                            for (unsigned int i = 0; i < dofs_per_cell; ++i)
                            {
                                cell_rhs(i) += 2.0 * deltat * scalar_product(neumann_loc_tensor, fe_face_values[velocity].value(i, q)) *
                                               fe_face_values.JxW(q);
                            }
                        }
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

    SolverControl solver_control(20000, 1e-7 * velocity_system_rhs.l2_norm());

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
    SolverControl solver_control(20000, 1e-7 * pressure_system_rhs.l2_norm());

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
    SolverControl solver_control(2000, 1e-7 * velocity_update_rhs.l2_norm());

    // Jacobi or SSOR
    TrilinosWrappers::PreconditionJacobi::AdditionalData data;
    data.omega = 0.7;
    data.n_sweeps = 5;

    TrilinosWrappers::PreconditionJacobi prec;
    prec.initialize(velocity_update_matrix, data);

    SolverCG<TrilinosWrappers::MPI::Vector> solver_cg(solver_control);

    solver_cg.solve(velocity_update_matrix, tmp, velocity_update_rhs, prec);

    std::cout << "Update CG iterations: " << solver_control.last_step() << std::endl;

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

    if constexpr (dim == 3)
    {

        exact_velocity3D.set_time(0.0);
        exact_pressure3D.set_time(0.0);

        {
            TrilinosWrappers::MPI::Vector tmp(locally_owned_velocity, MPI_COMM_WORLD);
            VectorTools::interpolate(dof_handler_velocity,
                                     exact_velocity3D,
                                     tmp);
            constraints_velocity.distribute(tmp);
            old_velocity = tmp;
            old_old_velocity = tmp;
            velocity_solution = tmp;
        }

        {
            TrilinosWrappers::MPI::Vector tmp(locally_owned_pressure, MPI_COMM_WORLD);
            VectorTools::interpolate(dof_handler_pressure,
                                     exact_pressure3D,
                                     tmp);
            constraints_pressure.distribute(tmp);
            pressure_solution = tmp;
            old_pressure = tmp;
        }

        {
            TrilinosWrappers::MPI::Vector tmp(locally_owned_pressure, MPI_COMM_WORLD);
            VectorTools::interpolate(dof_handler_pressure,
                                     Functions::ZeroFunction<dim>(),
                                        tmp);
            constraints_pressure.distribute(tmp);
            deltap = tmp;
        }
    }
    else if constexpr (dim == 2)
    {
        exact_velocity2D.set_time(0.0);
        exact_pressure2D.set_time(0.0);
        // Apply initial condition
        TrilinosWrappers::MPI::Vector tmp(locally_owned_velocity, MPI_COMM_WORLD);
        VectorTools::interpolate(dof_handler_velocity,
                                 exact_velocity2D,
                                 tmp);
        constraints_velocity.distribute(tmp);
        old_velocity = tmp;
        old_old_velocity = tmp;
        velocity_solution = tmp;

        TrilinosWrappers::MPI::Vector tmp_p(locally_owned_pressure, MPI_COMM_WORLD);
        VectorTools::interpolate(dof_handler_pressure,
                                 exact_pressure2D,
                                 tmp_p);
        constraints_pressure.distribute(tmp_p);
        deltap = tmp_p;
        pressure_solution = tmp_p;
    }

    // Possibly output initial
    output_results();

    while (time < T - 0.5 * deltat)
    {
        time += deltat;
        ++time_step;

        time = deltat * time_step;
        std::cout << "Time step " << time_step << " at t=" << time << std::endl;

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

        pressure_update(rotational);

        // if constexpr(dim == 3)
        // {
        //     exact_pressure3D.set_time(time);
        //      TrilinosWrappers::MPI::Vector tmp(locally_owned_pressure, MPI_COMM_WORLD);
        //     VectorTools::interpolate(dof_handler_pressure,
        //                              exact_pressure3D,
        //                              tmp);
        //     constraints_pressure.distribute(tmp);
        //     pressure_solution = tmp;
        // }

        output_results();
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

    std::string sub_dir_name = "outputs_";
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
double IncrementalChorinTemam<dim>::compute_error_velocity(const VectorTools::NormType &norm_type)
{
    FE_SimplexP<dim> fe_linear(degree_velocity);
    MappingFE mapping(fe_linear);
    const QGaussSimplex<dim> quadrature_error = QGaussSimplex<dim>(degree_velocity + 2);
    if constexpr (dim == 2)
        exact_velocity2D.set_time(time);
    if constexpr (dim == 3)
        exact_velocity3D.set_time(time);
    Vector<double> error_per_cell;
    if constexpr (dim == 2)
        VectorTools::integrate_difference(mapping,
                                          dof_handler_velocity,
                                          update_velocity_solution,
                                          exact_velocity2D,
                                          error_per_cell,
                                          quadrature_error,
                                          norm_type);
    if constexpr (dim == 3)
        VectorTools::integrate_difference(mapping,
                                          dof_handler_velocity,
                                          update_velocity_solution,
                                          exact_velocity3D,
                                          error_per_cell,
                                          quadrature_error,
                                          norm_type);
    const double error =
        VectorTools::compute_global_error(mesh, error_per_cell, norm_type);
    return error;
}

template <unsigned int dim>
double IncrementalChorinTemam<dim>::compute_error_pressure(const VectorTools::NormType &norm_type)
{
    FE_SimplexP<dim> fe_linear(degree_pressure);
    MappingFE mapping(fe_linear);
    const QGaussSimplex<dim> quadrature_error(degree_pressure + 2);
    if constexpr (dim == 2)
        exact_pressure2D.set_time(time);
    else if constexpr (dim == 3)
        exact_pressure3D.set_time(time);
    Vector<double> error_per_cell;
    if constexpr (dim == 2)
        VectorTools::integrate_difference(mapping,
                                          dof_handler_pressure,
                                          pressure_solution,
                                          exact_pressure2D,
                                          error_per_cell,
                                          quadrature_error,
                                          norm_type);
    else if constexpr (dim == 3)
        VectorTools::integrate_difference(mapping,
                                          dof_handler_pressure,
                                          pressure_solution,
                                          exact_pressure3D,
                                          error_per_cell,
                                          quadrature_error,
                                          norm_type);
    const double error =
        VectorTools::compute_global_error(mesh, error_per_cell, norm_type);
    return error;
}

template <unsigned int dim>
void IncrementalChorinTemam<dim>::update_buondary_conditions()
{
    if constexpr (dim == 2)
    {
        exact_velocity2D.set_time(time);

        constraints_velocity.clear();

        VectorTools::interpolate_boundary_values(dof_handler_velocity,
                                                 /*boundary_id=*/0,
                                                 exact_velocity2D,
                                                 constraints_velocity);

        VectorTools::interpolate_boundary_values(dof_handler_velocity,
                                                 /*boundary_id=*/1,
                                                 exact_velocity2D,
                                                 constraints_velocity);

        // Zero velocity (homogeneous Dirichlet) on boundary ID = 2:
        VectorTools::interpolate_boundary_values(dof_handler_velocity,
                                                 /*boundary_id=*/2,
                                                 exact_velocity2D,
                                                 constraints_velocity);

        // Zero velocity (homogeneous Dirichlet) on boundary ID = 3:
        VectorTools::interpolate_boundary_values(dof_handler_velocity,
                                                 /*boundary_id=*/3,
                                                 exact_velocity2D,
                                                 constraints_velocity);

        constraints_velocity.close();
    }
    else if constexpr (dim == 3)
    {
        exact_velocity3D.set_time(time);

        constraints_velocity.clear();

        VectorTools::interpolate_boundary_values(dof_handler_velocity,
                                                 /*boundary_id=*/0,
                                                 exact_velocity3D,
                                                 constraints_velocity);

        VectorTools::interpolate_boundary_values(dof_handler_velocity,
                                                 /*boundary_id=*/1,
                                                 exact_velocity3D,
                                                 constraints_velocity);

        // Zero velocity (homogeneous Dirichlet) on boundary ID = 3:
        VectorTools::interpolate_boundary_values(dof_handler_velocity,
                                                 /*boundary_id=*/2,
                                                 exact_velocity3D,
                                                 constraints_velocity);

        // Zero velocity (homogeneous Dirichlet) on boundary ID = 4:
        VectorTools::interpolate_boundary_values(dof_handler_velocity,
                                                 /*boundary_id=*/3,
                                                 exact_velocity3D,
                                                 constraints_velocity);

        VectorTools::interpolate_boundary_values(dof_handler_velocity,
                                                 /*boundary_id=*/5,
                                                 exact_velocity3D,
                                                 constraints_velocity);

        constraints_velocity.close();
    }
}

template <unsigned int dim>
void IncrementalChorinTemam<dim>::pressure_update(bool rotational)
{
    // Non-rotational variant: just p^{n+1} = p^n + deltap
    if (!rotational)
    {
        pressure_solution.add(deltap); // p^{n+1} = p^n + Δp
        return;
    }

    // -------------------------
    // Rotational variant:
    // p^{n+1} = p^n + Δp - ν * div(u_tilde)
    // We must project div(u_tilde) onto the same FE space as p.
    // -------------------------

    // 1) Build an L2 mass matrix for the pressure FE space
    TrilinosWrappers::SparseMatrix mass_matrix;
    {
        TrilinosWrappers::SparsityPattern dsp_p(locally_owned_pressure, MPI_COMM_WORLD);
        // Make sure we use the same constraints as for pressure (hanging nodes, BCs)
        DoFTools::make_sparsity_pattern(dof_handler_pressure,
                                        dsp_p,
                                        constraints_pressure,
                                        /*keep_constrained_dofs*/ false);
        dsp_p.compress();
        mass_matrix.reinit(dsp_p);
    }
    mass_matrix = 0.0;

    // 2) Build the RHS for the L2-projection of div(u_tilde)
    TrilinosWrappers::MPI::Vector rhs(locally_owned_pressure, MPI_COMM_WORLD);
    rhs = 0.0;

    const unsigned int quad_deg = std::max<unsigned int>(2u, fe_pressure.degree + 1u);
    QGaussSimplex<dim> quadrature_formula(quad_deg);

    FEValues<dim> fe_values_p(fe_pressure,
                              quadrature_formula,
                              update_values | update_quadrature_points | update_JxW_values);

    FEValues<dim> fe_values_v(fe_velocity,
                              quadrature_formula,
                              update_values | update_gradients | update_quadrature_points);

    const unsigned int n_q = quadrature_formula.size();
    const unsigned int dofs_per_cell_p = fe_pressure.dofs_per_cell;

    FullMatrix<double> cell_mass(dofs_per_cell_p, dofs_per_cell_p);
    Vector<double> cell_rhs(dofs_per_cell_p);
    std::vector<types::global_dof_index> local_indices(dofs_per_cell_p);

    // To read the velocity divergence:
    std::vector<double> local_div_u_tilde(n_q);

    // Extractor for the velocity field in the velocity FE:
    const FEValuesExtractors::Vector velocity_extract(0);

    // Loop over cells of the pressure and velocity DoFHandlers in parallel
    auto cell_p = dof_handler_pressure.begin_active();
    auto cell_v = dof_handler_velocity.begin_active();
    const auto end_p = dof_handler_pressure.end();

    for (; cell_p != end_p; ++cell_p, ++cell_v)
    {
        if (!cell_p->is_locally_owned())
            continue;

        fe_values_p.reinit(cell_p);
        fe_values_v.reinit(cell_v);

        // Get div(u_tilde) at quadrature points
        fe_values_v[velocity_extract].get_function_divergences(update_velocity_solution, local_div_u_tilde);

        cell_mass = 0.0;
        cell_rhs = 0.0;

        cell_p->get_dof_indices(local_indices);

        for (unsigned int q = 0; q < n_q; ++q)
        {
            const double div_val = local_div_u_tilde[q];
            const double JxW    = fe_values_p.JxW(q);

            for (unsigned int i = 0; i < dofs_per_cell_p; ++i)
            {
                const double phi_i = fe_values_p.shape_value(i, q);
                for (unsigned int j = 0; j < dofs_per_cell_p; ++j)
                {
                    const double phi_j = fe_values_p.shape_value(j, q);
                    // Mass matrix contribution
                    cell_mass(i, j) += (phi_i * phi_j) * JxW;
                }
                // RHS = ∫ div(u_tilde) * phi_i
                cell_rhs(i) += (div_val * phi_i) * JxW;
            }
        }

        // Add local contributions to global mass matrix & RHS
        constraints_pressure.distribute_local_to_global(cell_mass,
                                                        cell_rhs,
                                                        local_indices,
                                                        mass_matrix,
                                                        rhs);
    } // end cell loop

    mass_matrix.compress(VectorOperation::add);
    rhs.compress(VectorOperation::add);

    // 3) Solve M * (divProj) = rhs for the L2-projection of div(u_tilde)
    TrilinosWrappers::MPI::Vector div_projected(locally_owned_pressure, MPI_COMM_WORLD);

    {
        SolverControl solver_control(2000, 1e-12 * rhs.l2_norm());
        SolverCG<TrilinosWrappers::MPI::Vector> solver_cg(solver_control);

        TrilinosWrappers::PreconditionIC prec;
        prec.initialize(mass_matrix);

        solver_cg.solve(mass_matrix, div_projected, rhs, prec);
        constraints_pressure.distribute(div_projected);

        if (mpi_rank == 0)
            std::cout << "Pressure update CG iterations: " << solver_control.last_step() << std::endl;
    }

    // 4) Now apply the update:
    // p^{n+1} = p^n + Δp - ν * div_projected
    // (The user formula does not have a Δt factor, but add it if your scheme needs it.)
    pressure_solution.add(deltap);           // p^{n+1} = p^n + Δp
    pressure_solution.add(-nu, div_projected); // p^{n+1} -= ν * div(u_tilde)

    pressure_solution.update_ghost_values();
}


template class IncrementalChorinTemam<2>;
template class IncrementalChorinTemam<3>;
