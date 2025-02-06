#include "../include/SteadyNavierStokes.hpp"

// ===============================
// SteadyNavierStokes<dim> methods
// ===============================

template <int dim>
void SteadyNavierStokes<dim>::run_full_problem_pipeline(double& errorL2, double& errorH1)
{
    this->pcout << "===============================================" << std::endl;
    this->pcout << "Running full pipeline: Stokes -> NonLinearCorrection" << std::endl;
    this->pcout << "===============================================" << std::endl;

    // 1) Create a Stokes solver
    Stokes<dim> stokes_problem(this->mesh_file_name,
                               this->degree_velocity,
                               this->degree_pressure);

    // 2) Run Stokes solver
    stokes_problem.setup();
    stokes_problem.assemble();
    stokes_problem.solve();
    // stokes_problem.output();

    // 3) Retrieve Stokes solution
    TrilinosWrappers::MPI::BlockVector stokes_solution = stokes_problem.get_solution();

    // 4) Create NonLinearCorrection solver
    NonLinearCorrection<dim> incremental_stokes(stokes_problem);
    incremental_stokes.set_initial_conditions(stokes_solution);

    // 5) Run NonLinearCorrection solver
    incremental_stokes.setup();
    incremental_stokes.solve();
    // incremental_stokes.output();

    // 6) Compute errors
    errorL2 = incremental_stokes.compute_error(VectorTools::L2_norm, true);
    errorH1 = incremental_stokes.compute_error(VectorTools::H1_norm, true);

    this->pcout << "error L2: " << errorL2 << std::endl;
    this->pcout << "error H1: " << errorH1 << std::endl;
  
}

template <int dim>
void SteadyNavierStokes<dim>::setup()
{
  this->pcout << "Initializing the mesh" << std::endl;

  // Use a serial Triangulation to read the mesh from file .msh
  Triangulation<dim> mesh_serial;
  {
    GridIn<dim> grid_in;
    grid_in.attach_triangulation(mesh_serial);

    std::ifstream grid_in_file(this->mesh_file_name);
    AssertThrow(grid_in_file,
                ExcMessage("Could not open mesh file '" + this->mesh_file_name + "'"));

    // Read the .msh file into the serial Triangulation
    grid_in.read_msh(grid_in_file);
  }

  // Partition the mesh among MPI processes
  GridTools::partition_triangulation(this->mpi_size, mesh_serial);

  // Create a parallel Triangulation from the serial one
  const auto construction_data =
    TriangulationDescription::Utilities::create_description_from_triangulation(
      mesh_serial, MPI_COMM_WORLD);
  this->mesh.create_triangulation(construction_data);

  this->pcout << "  Number of elements = "
              << this->mesh.n_global_active_cells() << std::endl;
  this->pcout << "-----------------------------------------------" << std::endl;
}

// Generic methods for the base class, do nothing by default
template <int dim>
void SteadyNavierStokes<dim>::assemble()
{
}

template <int dim>
void SteadyNavierStokes<dim>::solve()
{
}

template <int dim>
void SteadyNavierStokes<dim>::output()
{
}

template <int dim>
std::string SteadyNavierStokes<dim>::get_output_directory()
{
  return "./";  
}

// ===============================
// Stokes<dim> methods
// ===============================

template <int dim>
void Stokes<dim>::setup()
{
  // First, call the base setup() to read/distribute the mesh:
  SteadyNavierStokes<dim>::setup();

  // Now proceed with the "Stokes" specifics:
  this->pcout << "Initializing the finite element space" << std::endl;

  // For velocity and pressure
  const FE_SimplexP<dim> fe_scalar_velocity(this->degree_velocity);
  const FE_SimplexP<dim> fe_scalar_pressure(this->degree_pressure);

  this->fe = std::make_unique<FESystem<dim>>(fe_scalar_velocity, dim,
                                             fe_scalar_pressure,   1);

  this->pcout << "  Velocity degree:           = "
              << fe_scalar_velocity.degree << std::endl;
  this->pcout << "  Pressure degree:           = "
              << fe_scalar_pressure.degree << std::endl;
  this->pcout << "  DoFs per cell              = "
              << this->fe->dofs_per_cell << std::endl;

  this->quadrature = std::make_unique<QGaussSimplex<dim>>(this->fe->degree + 1);
  this->pcout << "  Quadrature points per cell = "
              << this->quadrature->size() << std::endl;

  this->quadrature_face =
      std::make_unique<QGaussSimplex<dim - 1>>(this->fe->degree + 1);
  this->pcout << "  Quadrature points per face = "
              << this->quadrature_face->size() << std::endl;

  this->pcout << "-----------------------------------------------" << std::endl;

  // Initialize the DoF handler
  this->pcout << "Initializing the DoF handler" << std::endl;
  this->dof_handler.reinit(this->mesh);
  this->dof_handler.distribute_dofs(*this->fe);

  // Reorder velocity DoFs first, then pressure DoFs
  std::vector<unsigned int> block_component(dim + 1, 0);
  block_component[dim] = 1; // pressure is component "dim"
  DoFRenumbering::component_wise(this->dof_handler, block_component);

  // Owned + relevant DoFs
  this->locally_owned_dofs = this->dof_handler.locally_owned_dofs();
  DoFTools::extract_locally_relevant_dofs(this->dof_handler,
                                          this->locally_relevant_dofs);

  // Count how many DoFs are velocity vs. pressure
  const std::vector<types::global_dof_index> dofs_per_block =
    DoFTools::count_dofs_per_fe_block(this->dof_handler, block_component);
  const unsigned int n_u = dofs_per_block[0];
  const unsigned int n_p = dofs_per_block[1];

  this->block_owned_dofs.resize(2);
  this->block_relevant_dofs.resize(2);
  this->block_owned_dofs[0]    = this->locally_owned_dofs.get_view(0, n_u);
  this->block_owned_dofs[1]    = this->locally_owned_dofs.get_view(n_u, n_u + n_p);
  this->block_relevant_dofs[0] = this->locally_relevant_dofs.get_view(0, n_u);
  this->block_relevant_dofs[1] = this->locally_relevant_dofs.get_view(n_u, n_u + n_p);

  this->pcout << "  Number of DoFs: " << std::endl
              << "    velocity = " << n_u << std::endl
              << "    pressure = " << n_p << std::endl
              << "    total    = " << (n_u + n_p) << std::endl;

  this->pcout << "-----------------------------------------------" << std::endl;

  // Initialize the linear system
  {
    this->pcout << "Initializing the linear system" << std::endl;
    this->pcout << "  Initializing the sparsity pattern" << std::endl;

    // Make the system matrix sparsity
    Table<2, DoFTools::Coupling> coupling(dim + 1, dim + 1);
    for (unsigned int c = 0; c < dim + 1; ++c)
      for (unsigned int d = 0; d < dim + 1; ++d)
      {
        // For the system matrix, everything except p-p block is "always"
        if (c == dim && d == dim)
          coupling[c][d] = DoFTools::none;
        else
          coupling[c][d] = DoFTools::always;
      }

    TrilinosWrappers::BlockSparsityPattern sparsity(this->block_owned_dofs,
                                                    MPI_COMM_WORLD);
    DoFTools::make_sparsity_pattern(this->dof_handler,
                                    coupling,
                                    sparsity);
    sparsity.compress();

    // Sparsity pattern for the pressure mass matrix (only p-p block)
    for (unsigned int c = 0; c < dim + 1; ++c)
      for (unsigned int d = 0; d < dim + 1; ++d)
      {
        if (c == dim && d == dim)
          coupling[c][d] = DoFTools::always;
        else
          coupling[c][d] = DoFTools::none;
      }

    TrilinosWrappers::BlockSparsityPattern sparsity_pressure_mass(
        this->block_owned_dofs, MPI_COMM_WORLD);
    DoFTools::make_sparsity_pattern(this->dof_handler,
                                    coupling,
                                    sparsity_pressure_mass);
    sparsity_pressure_mass.compress();

    this->pcout << "  Initializing the matrices" << std::endl;
    this->system_matrix.reinit(sparsity);
    this->pressure_mass.reinit(sparsity_pressure_mass);

    this->pcout << "  Initializing the system right-hand side" << std::endl;
    this->system_rhs.reinit(this->block_owned_dofs, MPI_COMM_WORLD);

    this->pcout << "  Initializing the solution vector" << std::endl;
    this->solution_owned.reinit(this->block_owned_dofs, MPI_COMM_WORLD);
    this->solution.reinit(this->block_owned_dofs,
                          this->block_relevant_dofs,
                          MPI_COMM_WORLD);
  }
}

template <int dim>
void Stokes<dim>::assemble()
{
  this->pcout  << "===============================================" << std::endl;
  this->pcout << "Assembling the system" << std::endl;

  const unsigned int dofs_per_cell = this->fe->dofs_per_cell;
  const unsigned int n_q           = this->quadrature->size();
  const unsigned int n_q_face      = this->quadrature_face->size();

  FEValues<dim> fe_values(*this->fe, *this->quadrature,
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

  FEFaceValues<dim> fe_face_values(*this->fe, *this->quadrature_face,
                                   update_values | update_normal_vectors | update_quadrature_points |  update_JxW_values);

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> cell_pressure_mass_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

  this->system_matrix = 0.0;
  this->system_rhs    = 0.0;
  this->pressure_mass = 0.0;

  Vector<double> neumann_loc(dim + 1);
  Tensor<1, dim> neumann_loc_tensor;

  FEValuesExtractors::Vector velocity(0);
  FEValuesExtractors::Scalar pressure(dim);

  for (const auto &cell : this->dof_handler.active_cell_iterators())
  {
    if (!cell->is_locally_owned())
      continue;

    fe_values.reinit(cell);
    cell_matrix               = 0.0;
    cell_rhs                  = 0.0;
    cell_pressure_mass_matrix = 0.0;

    for (unsigned int q = 0; q < n_q; ++q)
    {
      Vector<double> forcing_loc(dim);
      this->forcing_term.vector_value(fe_values.quadrature_point(q), forcing_loc);

      Tensor<1, dim> forcing_tensor;
      for (unsigned int d = 0; d < dim; ++d)
        forcing_tensor[d] = forcing_loc[d];

      for (unsigned int i = 0; i < dofs_per_cell; ++i)
      {
        for (unsigned int j = 0; j < dofs_per_cell; ++j)
        {
          cell_matrix(i, j) += this->nu 
                               * scalar_product(fe_values[velocity].gradient(i, q),
                                                 fe_values[velocity].gradient(j, q))
                               * fe_values.JxW(q);

          cell_matrix(i, j) -= fe_values[velocity].divergence(i, q)
                               * fe_values[pressure].value(j, q)
                               * fe_values.JxW(q);

          cell_matrix(i, j) -= fe_values[velocity].divergence(j, q)
                               * fe_values[pressure].value(i, q)
                               * fe_values.JxW(q);

          cell_pressure_mass_matrix(i, j) +=
                               fe_values[pressure].value(i, q)
                               * fe_values[pressure].value(j, q)
                               / this->nu 
                               * fe_values.JxW(q);
        }

        // Forcing
        cell_rhs(i) += scalar_product(forcing_tensor,
                                      fe_values[velocity].value(i, q))
                       * fe_values.JxW(q);
      }
    }

    if (cell->at_boundary())
    {
      for (unsigned int f = 0; f < cell->n_faces(); ++f)
      {
        if (cell->face(f)->at_boundary() && (cell->face(f)->boundary_id() == 0))
        {
          fe_face_values.reinit(cell, f);
          for (unsigned int q = 0; q < n_q_face; ++q)
          {

          this->neumann_function.vector_value(fe_face_values.quadrature_point(q), neumann_loc);

          for (unsigned int d = 0; d < dim; ++d)
              neumann_loc_tensor[d] = neumann_loc[d];

          for (unsigned int i = 0; i < dofs_per_cell; ++i)
          {
              cell_rhs(i) += scalar_product(neumann_loc_tensor, fe_face_values[velocity].value(i, q)) *
                              fe_face_values.JxW(q);
          }
         }
       }
     }
    }

    cell->get_dof_indices(dof_indices);
    this->system_matrix.add(dof_indices, cell_matrix);
    this->system_rhs.add(dof_indices, cell_rhs);
    this->pressure_mass.add(dof_indices, cell_pressure_mass_matrix);
  }

  this->system_matrix.compress(VectorOperation::add);
  this->system_rhs.compress(VectorOperation::add);
  this->pressure_mass.compress(VectorOperation::add);

  // Dirichlet boundary conditions
  {
    std::map<types::global_dof_index, double> boundary_values;
    std::map<types::boundary_id, const Function<dim> *> boundary_functions;

    Functions::ZeroFunction<dim> zero_function(dim + 1);

    // Dirichlet Boundary Conditions
      boundary_functions[1] = &this->exact_solution;         // Outlet
      boundary_functions[2] = &this->exact_solution;         // Walls
      boundary_functions[3] = &this->exact_solution;         // Obstacle
    
    if constexpr (dim == 2)
    {
      // Interpolate boundary values into constraints (only velocity components)
      VectorTools::interpolate_boundary_values(
      this->dof_handler,
      boundary_functions,
      boundary_values,
      ComponentMask({true,true, false ,false}));
    }
    else if constexpr (dim == 3)
    {
      // Interpolate boundary values into constraints (only velocity components)
      VectorTools::interpolate_boundary_values(
      this->dof_handler,
      boundary_functions,
      boundary_values,
      ComponentMask({true,true, true,false}));
    }

    // Apply them
    MatrixTools::apply_boundary_values(boundary_values,
                                       this->system_matrix,
                                       this->solution_owned,
                                       this->system_rhs,
                                       false);
                                       
  }
}


template <int dim>
void Stokes<dim>::solve()
{
  this->pcout << "===============================================" << std::endl;

  SolverControl solver_control(2000, 1e-6 * this->system_rhs.l2_norm());
  SolverGMRES<TrilinosWrappers::MPI::BlockVector> solver(solver_control);

  // Example: use a block triangular preconditioner
  typename SteadyNavierStokes<dim>::PreconditionBlockTriangularStokes preconditioner;
  preconditioner.initialize(this->system_matrix.block(0,0),
                            this->pressure_mass.block(1,1),
                            this->system_matrix.block(1,0));

  this->pcout << "Solving the linear system" << std::endl;
  solver.solve(this->system_matrix, this->solution_owned,
               this->system_rhs, preconditioner);
  this->pcout << "  " << solver_control.last_step() << " GMRES iterations"
              << std::endl;

  // Distribute to the fully relevant solution
  this->solution = this->solution_owned;

}


template <int dim>
void Stokes<dim>::output()
{
  this->pcout << "===============================================" << std::endl;

  DataOut<dim> data_out;

  // velocity = vector, pressure = scalar
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
    data_component_interpretation(dim,
      DataComponentInterpretation::component_is_part_of_vector);
  data_component_interpretation.push_back(
      DataComponentInterpretation::component_is_scalar);

  // For 2D, we have velocity_x, velocity_y, then pressure
  std::vector<std::string> names;
  for (unsigned int i = 0; i < dim; ++i)
    names.push_back("velocity");
  names.push_back("pressure");

  data_out.add_data_vector(this->dof_handler,
                           this->solution,
                           names,
                           data_component_interpretation);

  // Add partitioning info
  std::vector<unsigned int> partition_int(this->mesh.n_active_cells());
  GridTools::get_subdomain_association(this->mesh, partition_int);
  const Vector<double> partitioning(partition_int.begin(), partition_int.end());
  data_out.add_data_vector(partitioning, "partitioning");

  data_out.build_patches();

  std::string numProcessors = std::to_string(this->mpi_size);
  numProcessors += (this->mpi_size == 1) ? "_processor" : "_processors";

  const std::string output_file_name = "output-Stokes-" + numProcessors;
  data_out.write_vtu_with_pvtu_record(this->get_output_directory(),
                                      output_file_name,
                                      0,
                                      MPI_COMM_WORLD);

  this->pcout << "Output written to " << output_file_name << std::endl;
  this->pcout << "===============================================" << std::endl;
}

template <int dim>
std::string Stokes<dim>::get_output_directory()
{
    namespace fs = std::filesystem;

    // 1) Ensure top-level "outputs/" exists
    if (!fs::exists("outputs"))
        fs::create_directory("outputs");

    // 2) Create a subdirectory specific to "SteadyNavierStokes"
    if (!fs::exists("outputs/SteadyNavierStokes"))
        fs::create_directory("outputs/SteadyNavierStokes");

    // 3) Create a subdirectory specific to "Stokes"
    if (!fs::exists("outputs/SteadyNavierStokes/Stokes"))
        fs::create_directory("outputs/SteadyNavierStokes/Stokes");

    // 4) Further subdivide by Reynolds number (or any relevant parameter)
    const std::string sub_dir_name =
        "outputs_reynolds_" + std::to_string(static_cast<int>(1.0 / this->nu));

    fs::path sub_dir_path = 
        fs::path("outputs/SteadyNavierStokes/Stokes") / sub_dir_name;

    if (!fs::exists(sub_dir_path))
        fs::create_directory(sub_dir_path);

    // Return the absolute string path to the new directory
    return sub_dir_path.string();
}


// ===============================
// NonLinearCorrection<dim> methods
// ===============================

template <int dim>
void NonLinearCorrection<dim>::setup()
{
  // We do NOT read the mesh again because it was already copied in the construtor.
  // We do want to initialize the FE system, quadrature, etc.

  this->pcout << "Initializing the finite element space" << std::endl;

  const FE_SimplexP<dim> fe_scalar_velocity(this->degree_velocity);
  const FE_SimplexP<dim> fe_scalar_pressure(this->degree_pressure);
  this->fe = std::make_unique<FESystem<dim>>(fe_scalar_velocity, dim,
                                             fe_scalar_pressure,   1);

  this->pcout << "  Velocity degree:           = "
              << fe_scalar_velocity.degree << std::endl;
  this->pcout << "  Pressure degree:           = "
              << fe_scalar_pressure.degree << std::endl;
  this->pcout << "  DoFs per cell              = "
              << this->fe->dofs_per_cell << std::endl;

  this->quadrature = std::make_unique<QGaussSimplex<dim>>(this->fe->degree + 1);
  this->pcout << "  Quadrature points per cell = "
              << this->quadrature->size() << std::endl;

  this->quadrature_face =
      std::make_unique<QGaussSimplex<dim - 1>>(this->fe->degree + 1);
  this->pcout << "  Quadrature points per face = "
              << this->quadrature_face->size() << std::endl;
  this->pcout << "-----------------------------------------------" << std::endl;

  // Initialize the DoF handler
  this->pcout << "Initializing the DoF handler" << std::endl;
  this->dof_handler.reinit(this->mesh);
  this->dof_handler.distribute_dofs(*this->fe);

  std::vector<unsigned int> block_component(dim + 1, 0);
  block_component[dim] = 1;
  DoFRenumbering::component_wise(this->dof_handler, block_component);

  this->locally_owned_dofs = this->dof_handler.locally_owned_dofs();
  DoFTools::extract_locally_relevant_dofs(this->dof_handler,
                                          this->locally_relevant_dofs);

  std::vector<types::global_dof_index> dofs_per_block =
      DoFTools::count_dofs_per_fe_block(this->dof_handler, block_component);
  const unsigned int n_u = dofs_per_block[0];
  const unsigned int n_p = dofs_per_block[1];

  this->block_owned_dofs.resize(2);
  this->block_relevant_dofs.resize(2);
  this->block_owned_dofs[0]    = this->locally_owned_dofs.get_view(0, n_u);
  this->block_owned_dofs[1]    = this->locally_owned_dofs.get_view(n_u, n_u + n_p);
  this->block_relevant_dofs[0] = this->locally_relevant_dofs.get_view(0, n_u);
  this->block_relevant_dofs[1] = this->locally_relevant_dofs.get_view(n_u, n_u + n_p);

  this->pcout << "  Number of DoFs: " << std::endl
              << "    velocity = " << n_u << std::endl
              << "    pressure = " << n_p << std::endl
              << "    total    = " << (n_u + n_p) << std::endl;
  this->pcout << "-----------------------------------------------" << std::endl;

  // Initialize constraints
  {
    constraints.clear();
    // Define boundary conditions based on dimension
    std::map<types::boundary_id, const Function<dim> *> boundary_functions;
    Functions::ZeroFunction<dim> zero_function(dim + 1);

    // Dirichlet Boundary Conditions
      boundary_functions[1] = &this->exact_solution;         // Outlet
      boundary_functions[2] = &this->exact_solution;         // Walls
      boundary_functions[3] = &this->exact_solution;         // Obstacle

    if constexpr (dim == 2)
    {
      // Interpolate boundary values into constraints (only velocity components)
      VectorTools::interpolate_boundary_values(this->dof_handler,
                                             boundary_functions,
                                             constraints,
                                             ComponentMask({true,true,false,false}));
    }
    else if constexpr (dim == 3)
    {
      // Interpolate boundary values into constraints (only velocity components)
      VectorTools::interpolate_boundary_values(this->dof_handler,
                                             boundary_functions,
                                             constraints,
                                             ComponentMask({true,true,true,false}));
    }

    constraints.close();
  }

  // Initialize system
  this->pcout << "Initializing the linear system" << std::endl;
  this->pcout << "Initializing the sparsity pattern" << std::endl;

  Table<2, DoFTools::Coupling> coupling(dim + 1, dim + 1);
  for (unsigned int c = 0; c < dim + 1; ++c)
    for (unsigned int d = 0; d < dim + 1; ++d)
      coupling[c][d] = DoFTools::always; // we assume all couplings

  TrilinosWrappers::BlockSparsityPattern sparsity(this->block_owned_dofs,
                                                  MPI_COMM_WORLD);
  DoFTools::make_sparsity_pattern(this->dof_handler,
                                  coupling,
                                  sparsity,
                                  constraints,
                                  false);
  sparsity.compress();

  // Pressure mass matrix pattern (only p-p block)
  for (unsigned int c = 0; c < dim + 1; ++c)
    for (unsigned int d = 0; d < dim + 1; ++d)
    {
      if (c == dim && d == dim)
        coupling[c][d] = DoFTools::always;
      else
        coupling[c][d] = DoFTools::none;
    }

  TrilinosWrappers::BlockSparsityPattern sparsity_pressure_mass(
      this->block_owned_dofs, MPI_COMM_WORLD);
  DoFTools::make_sparsity_pattern(this->dof_handler,
                                  coupling,
                                  sparsity_pressure_mass,
                                  constraints,
                                  false);
  sparsity_pressure_mass.compress();

  this->system_matrix.reinit(sparsity);
  this->pressure_mass.reinit(sparsity_pressure_mass);

  this->system_rhs.reinit(this->block_owned_dofs, MPI_COMM_WORLD);
  this->solution_owned.reinit(this->block_owned_dofs, MPI_COMM_WORLD);
  this->solution.reinit(this->block_owned_dofs,
                        this->block_relevant_dofs,
                        MPI_COMM_WORLD);
}


template <int dim>
void NonLinearCorrection<dim>::assemble()
{
  const unsigned int dofs_per_cell = this->fe->dofs_per_cell;
  const unsigned int n_q           = this->quadrature->size();
  const unsigned int n_q_face      = this->quadrature_face->size();

  FEValues<dim> fe_values(*this->fe, *this->quadrature,
                          update_values | update_gradients |
                          update_quadrature_points | update_JxW_values);

  FEFaceValues<dim> fe_face_values(*this->fe, *this->quadrature_face,
                                   update_values | update_normal_vectors | update_quadrature_points |update_JxW_values);

  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);
  FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     local_rhs(dofs_per_cell);

  // Reset system contributions
  this->system_matrix = 0.0;
  this->system_rhs    = 0.0;

  // Temporary arrays for old solution evaluations
  std::vector<Tensor<1, dim>> previous_velocity_values(n_q);
  std::vector<Tensor<2, dim>> previous_velocity_gradients(n_q);
  std::vector<double>         previous_velocity_divergence(n_q);
  std::vector<double>         previous_pressure_values(n_q);

  // Predefine shape function evaluations
  std::vector<double>         div_phi_u(dofs_per_cell);
  std::vector<Tensor<1, dim>> phi_u(dofs_per_cell);
  std::vector<Tensor<2, dim>> grad_phi_u(dofs_per_cell);
  std::vector<double>         phi_p(dofs_per_cell);

  for (const auto &cell : this->dof_handler.active_cell_iterators())
  {
    if (!cell->is_locally_owned())
      continue;

    fe_values.reinit(cell);
    local_matrix = 0.0;
    local_rhs    = 0.0;

    Vector<double> neumann_loc(dim + 1);
    Tensor<1, dim> neumann_loc_tensor;

    // Evaluate the old solution at quadrature points
    fe_values[u_k].get_function_values(this->solution_old, previous_velocity_values);
    fe_values[u_k].get_function_gradients(this->solution_old, previous_velocity_gradients);
    fe_values[p_k].get_function_values(this->solution_old, previous_pressure_values);
    fe_values[u_k].get_function_divergences(this->solution_old, previous_velocity_divergence);

    for (unsigned int q = 0; q < n_q; ++q)
    {      
      Vector<double> forcing_loc(dim);
      this->forcing_term.vector_value(fe_values.quadrature_point(q), forcing_loc);
      Tensor<1, dim> forcing_tensor;
      for (unsigned int d = 0; d < dim; ++d)
        forcing_tensor[d] = forcing_loc[d];

      for (unsigned int k = 0; k < dofs_per_cell; ++k)
      {
        div_phi_u[k]  = fe_values[u_k].divergence(k, q);
        grad_phi_u[k] = fe_values[u_k].gradient(k, q);
        phi_u[k]      = fe_values[u_k].value(k, q);
        phi_p[k]      = fe_values[p_k].value(k, q);
      }

      // Build local contributions (viscous, convection, and divergence terms)
      for (unsigned int i = 0; i < dofs_per_cell; ++i)
      {
        for (unsigned int j = 0; j < dofs_per_cell; ++j)
        {
          local_matrix(i, j) += this->nu *
                                scalar_product(grad_phi_u[i], grad_phi_u[j]) *
                                fe_values.JxW(q);

          local_matrix(i, j) += phi_u[j] *
                                (transpose(previous_velocity_gradients[q]) * phi_u[i])
                                * fe_values.JxW(q);

          local_matrix(i, j) += (previous_velocity_values[q] *
                                transpose(grad_phi_u[j])) *
                                phi_u[i] *
                                fe_values.JxW(q);

          local_matrix(i, j) += 0.5 * previous_velocity_divergence[q] *  // Skew Symmetric Term
                                phi_u[j] *
                                phi_u[i] *
                                fe_values.JxW(q);

          local_matrix(i, j) -= phi_p[j] * div_phi_u[i] * fe_values.JxW(q);
          local_matrix(i, j) -= phi_p[i] * div_phi_u[j] * fe_values.JxW(q);
        }

        local_rhs(i) += (previous_velocity_values[q] *
                         transpose(previous_velocity_gradients[q]) *
                         phi_u[i]) * fe_values.JxW(q);

        local_rhs(i) += scalar_product(forcing_tensor, phi_u[i]) * fe_values.JxW(q);
      }
    }
    // --- Neumann BC application (on boundary_id == 0) ---
if (cell->at_boundary())
    {
      for (unsigned int f = 0; f < cell->n_faces(); ++f)
      {
        if (cell->face(f)->at_boundary() && (cell->face(f)->boundary_id() == 0))
        {
          fe_face_values.reinit(cell, f);
          for (unsigned int q = 0; q < n_q_face; ++q)
          {

          this->neumann_function.vector_value(fe_face_values.quadrature_point(q), neumann_loc);

          for (unsigned int d = 0; d < dim; ++d)
              neumann_loc_tensor[d] = neumann_loc[d];

          for (unsigned int i = 0; i < dofs_per_cell; ++i)
          {
              local_rhs(i) += scalar_product(neumann_loc_tensor, fe_face_values[u_k].value(i, q)) *
                              fe_face_values.JxW(q);
          }
         }
       }
     }
    }
    // -----------------------------------------------------

    cell->get_dof_indices(dof_indices);
    constraints.distribute_local_to_global(local_matrix,
                                           local_rhs,
                                           dof_indices,
                                           this->system_matrix,
                                           this->system_rhs);
  
  }
  this->system_matrix.compress(VectorOperation::add);
  this->system_rhs.compress(VectorOperation::add);
}



template <int dim>
void NonLinearCorrection<dim>::solve()
{
  for (unsigned int iteration = 0; iteration < this->maxIter; ++iteration)
  {
    // 1) Assemble system using the current "solution_old"
    this->assemble();

    
    double local_sum = 0.0;
    for (unsigned int k = 0; k < this->solution_old.size(); ++k)
      local_sum += this->solution_old(k) * this->solution_old(k);

    double global_sum = 0.0;
    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE,
                  MPI_SUM, MPI_COMM_WORLD);
    double solution_norm = std::sqrt(global_sum);

    if (this->mpi_rank == 0)
      std::cout << "L2 norm of the solution = " << solution_norm << std::endl;
    
    
    // 2) Prepare for the solve: start from the old solution (not zero)
    this->solution_owned = this->solution_old;
    constraints.distribute(this->solution_owned); // apply boundary constraints

    // 3) Solve with a chosen iterative method
    SolverControl solver_control(/*max steps*/ 20000000, /*tol*/ 5e-7);
    SolverGMRES<TrilinosWrappers::MPI::BlockVector> solver(solver_control);

    // Example: identity preconditioner
    typename SteadyNavierStokes<dim>::PreconditionIdentity preconditioner;
    // constraints.set_zero(this->solution_owned);

    solver.solve(this->system_matrix,
                 this->solution_owned,
                 this->system_rhs,
                 preconditioner);

    constraints.distribute(this->solution_owned);
    // Enforce constraints on the final solution

    this->pcout << "  " << solver_control.last_step()
                << " GMRES iterations" << std::endl;

    // 4) Copy this final solution to "this->solution"
    this->solution = this->solution_owned;

    // 5) Compute the update difference = solution - solution_old
    this->new_res.reinit(this->solution);
    this->new_res = this->solution;
    this->new_res.sadd(/*a=*/1.0, /*b=*/-1.0, this->solution_old);

    // 6) Compute the L2 norm of that update
    local_sum = 0.0;
    for (unsigned int k = 0; k < this->new_res.size(); ++k)
      local_sum += this->new_res(k) * this->new_res(k);

    global_sum = 0.0;
    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE,
                  MPI_SUM, MPI_COMM_WORLD);
    double update_norm = std::sqrt(global_sum);

    if (this->mpi_rank == 0)
      std::cout << "L2 norm of the update = " << update_norm << std::endl;

    // 7) Update the old solution for the next iteration
    this->solution_old = this->solution;

    if (this->mpi_rank == 0)
      std::cout << "Iteration " << iteration << " completed." << std::endl;

    // 8) Check for convergence
    if (update_norm < update_tol)
      break;
  }
}


template <int dim>
void NonLinearCorrection<dim>::output()
{
  this->pcout << "===============================================" << std::endl;

  DataOut<dim> data_out;

  std::vector<DataComponentInterpretation::DataComponentInterpretation>
    data_component_interpretation(dim,
      DataComponentInterpretation::component_is_part_of_vector);
  data_component_interpretation.push_back(
      DataComponentInterpretation::component_is_scalar);

  std::vector<std::string> names;
  for (unsigned int i = 0; i < dim; ++i)
    names.push_back("velocity");
  names.push_back("pressure");

  data_out.add_data_vector(this->dof_handler,
                           this->solution,
                           names,
                           data_component_interpretation);

  // Partition info
  std::vector<unsigned int> partition_int(this->mesh.n_active_cells());
  GridTools::get_subdomain_association(this->mesh, partition_int);
  const Vector<double> partitioning(partition_int.begin(), partition_int.end());
  data_out.add_data_vector(partitioning, "partitioning");

  data_out.build_patches();

  std::string numProcessors = std::to_string(this->mpi_size);
  numProcessors += (this->mpi_size == 1) ? "_processor" : "_processors";

  const std::string output_file_name = "output-nonLinearCorrection-" + numProcessors;
  data_out.write_vtu_with_pvtu_record(this->get_output_directory(),
                                      output_file_name,
                                      0,
                                      MPI_COMM_WORLD);

  this->pcout << "Output written to " << output_file_name << std::endl;
  this->pcout << "===============================================" << std::endl;
}

template <int dim>
std::string NonLinearCorrection<dim>::get_output_directory()
{
    namespace fs = std::filesystem;

    // 1) Ensure top-level "outputs/" exists
    if (!fs::exists("outputs"))
        fs::create_directory("outputs");

    // 2) Create a subdirectory specific to "SteadyNavierStokes"
    if (!fs::exists("outputs/SteadyNavierStokes"))
        fs::create_directory("outputs/SteadyNavierStokes");
    
    // 3) Create a subdirectory specific to "nonLinearCorrection"
    if (!fs::exists("outputs/SteadyNavierStokes/nonLinearCorrection"))
        fs::create_directory("outputs/SteadyNavierStokes/nonLinearCorrection");

    // 4) Further subdivide by Reynolds number (or any relevant parameter)
    const std::string sub_dir_name = 
        "outputs_reynolds_" + std::to_string(static_cast<int>(1.0/this->nu));

    fs::path sub_dir_path = 
        fs::path("outputs/SteadyNavierStokes/nonLinearCorrection") / sub_dir_name;

    if (!fs::exists(sub_dir_path))
        fs::create_directory(sub_dir_path);

    // Return the absolute string path to the new directory
    return sub_dir_path.string();
}



template <int dim>
double SteadyNavierStokes<dim>::compute_error(const VectorTools::NormType &norm_type, bool velocity) 
{
  FE_SimplexP<dim> fe_mapping(1);
  MappingFE mapping(fe_mapping);

  // First we compute the norm on each element, and store it in a vector.
  Vector<double> error_per_cell(this->mesh.n_active_cells());

  if (velocity) {
    // The error is an integral, and we approximate that integral using a
    // quadrature formula. To make sure we are accurate enough, we use a
    // quadrature formula with one node more than what we used in
    // assembly.
    QGaussSimplex<dim> quadrature_error(this->degree_velocity + 2);

    // Source: https://www.dealii.org/current/doxygen/deal.II/step_20.html.
    ComponentSelectFunction<dim> mask(std::make_pair(0, dim), dim + 1);

    VectorTools::integrate_difference(
        mapping, this->dof_handler, this->solution, this->exact_solution.exact_velocity,
        error_per_cell, quadrature_error, norm_type, &mask);
    
  } else {
    // Do the same for the pressure.
    QGaussSimplex<dim> quadrature_error(this->degree_pressure + 2);
    ComponentSelectFunction<dim> mask(dim, dim + 1);
    VectorTools::integrate_difference(mapping, this->dof_handler, this->solution,
                                      this->exact_solution.exact_pressure, error_per_cell,
                                      quadrature_error, norm_type, &mask);
  }

  // Then, we add out all the cells.
  const double error =
      VectorTools::compute_global_error(this->mesh, error_per_cell, norm_type);

  return error;
}




// ===============================
// Explicit Instantiations
// ===============================

template class SteadyNavierStokes<2>;
template class Stokes<2>;
template class NonLinearCorrection<2>;
