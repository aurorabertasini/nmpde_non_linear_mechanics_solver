#ifndef STEADYNAVIERSTOKES_HPP
#define STEADYNAVIERSTOKES_HPP

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/function.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/tensor.h>

#include <deal.II/distributed/fully_distributed_tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_values_extractors.h>
#include <deal.II/fe/mapping_fe.h>

#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_refinement.h>

#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/trilinos_block_sparse_matrix.h>
#include <deal.II/lac/trilinos_parallel_block_vector.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/full_matrix.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/error_estimator.h>

#include <fstream>
#include <filesystem>
#include <iostream>
#include <mpi.h>

using namespace dealii;

/**
 * Base class: SteadyNavierStokes
 *
 * Templated on <int dim>
 * Holds:
 *  - Mesh & polynomial degrees
 *  - Constants Re, H, D, uMax, uMean, p_out, nu
 *  - One method run_full_problem_pipeline() that instantiates
 *    and runs Stokes -> IncrementalStokes in a single workflow.
 */
template <int dim>
class SteadyNavierStokes
{
public:
  /**
   * Main constructor taking user inputs.
   */
  SteadyNavierStokes(const std::string &mesh_file_name_in,
                     const unsigned int degree_velocity_in,
                     const unsigned int degree_pressure_in,
                     const double       Re_in)
    : mesh_file_name(mesh_file_name_in)
    , degree_velocity(degree_velocity_in)
    , degree_pressure(degree_pressure_in)
    , H(0.41)       // Fixed by geometry
    , D(0.1)        // Fixed by geometry
    , uMax(dim == 2 ? 0.3 : 0.45) // Fixed by problem
    , uMean(dim == 2 ? 2./3. * uMax : 4./9. * uMax) // Fixed by problem
    , Re(Re_in)     // User input
    , p_out(0.0)    // Fixed by problem
    , nu(uMean * D / Re)  // Fixed by problem
    , mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD))
    , mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
    , pcout(std::cout, mpi_rank == 0)
    , mesh(MPI_COMM_WORLD)
    , dof_handler(mesh)
    , forcing_term()
    , inlet_velocity(H, uMax)
  {
  }

  // Disallow copy construction for clarity.
  SteadyNavierStokes(const SteadyNavierStokes<dim> &) = delete;
  virtual ~SteadyNavierStokes() = default;

  /**
   * A single method that instantiates a Stokes solver,
   * runs it, then instantiates an IncrementalStokes solver,
   * sets initial conditions, runs it, and finally
   * calls compute_lift_drag().
   */
  void run_full_problem_pipeline();

  // -----------------------------------------------------------
  // Virtual methods for the typical solve steps
  // -----------------------------------------------------------
  virtual void setup();
  virtual void assemble();
  virtual void solve();
  virtual void output();
  virtual std::string get_output_directory();

  // -----------------------------------------------------------
  // Public getters 
  // -----------------------------------------------------------
  const std::string & get_mesh_file_name() const
  {
    return mesh_file_name;
  }

  unsigned int get_degree_velocity() const
  {
    return degree_velocity;
  }

  unsigned int get_degree_pressure() const
  {
    return degree_pressure;
  }

  double get_uMax() const
  {
    return uMax;
  }

  double get_H() const
  {
    return H;
  }

  double get_p_out() const
  {
    return p_out;
  }

  double get_Re() const
  {
    return Re;
  }

  // Provide access to the mesh
  const parallel::fullydistributed::Triangulation<dim> & get_mesh() const
  {
    return mesh;
  }

  // -----------------------------------------------------------
  // Custom Preconditioners
  // -----------------------------------------------------------

  /**
   * A trivial identity preconditioner that returns dst = src.
   */
  class PreconditionIdentity
  {
  public:
    void vmult(TrilinosWrappers::MPI::BlockVector       &dst,
               const TrilinosWrappers::MPI::BlockVector &src) const
    {
      dst = src;
    }
  };

  /**
   * A block-diagonal preconditioner for (velocity, pressure).
   */
  class PreconditionBlockDiagonal
  {
  public:
    void initialize(const TrilinosWrappers::SparseMatrix &velocity_stiffness_,
                    const TrilinosWrappers::SparseMatrix &pressure_mass_)
    {
      velocity_stiffness = &velocity_stiffness_;
      pressure_mass      = &pressure_mass_;

      preconditioner_velocity.initialize(velocity_stiffness_);
      preconditioner_pressure.initialize(pressure_mass_);
    }

    void vmult(TrilinosWrappers::MPI::BlockVector       &dst,
               const TrilinosWrappers::MPI::BlockVector &src) const
    {
      // Velocity block
      {
        SolverControl                           solver_control_velocity(100000,
                                               1e-2 * src.block(0).l2_norm());
        SolverCG<TrilinosWrappers::MPI::Vector> solver_cg_velocity(solver_control_velocity);
        solver_cg_velocity.solve(*velocity_stiffness,
                                 dst.block(0),
                                 src.block(0),
                                 preconditioner_velocity);
      }

      // Pressure block
      {
        SolverControl                           solver_control_pressure(1000,
                                               1e-2 * src.block(1).l2_norm());
        SolverCG<TrilinosWrappers::MPI::Vector> solver_cg_pressure(solver_control_pressure);
        solver_cg_pressure.solve(*pressure_mass,
                                 dst.block(1),
                                 src.block(1),
                                 preconditioner_pressure);
      }
    }

  protected:
    const TrilinosWrappers::SparseMatrix *velocity_stiffness = nullptr;
    const TrilinosWrappers::SparseMatrix *pressure_mass      = nullptr;

    TrilinosWrappers::PreconditionILU preconditioner_velocity;
    TrilinosWrappers::PreconditionILU preconditioner_pressure;
  };

  /**
   * A block-triangular preconditioner for the Stokes system that
   * solves velocity first, then corrects pressure.
   */
  class PreconditionBlockTriangularStokes
  {
  public:
    void initialize(const TrilinosWrappers::SparseMatrix &velocity_stiffness_,
                    const TrilinosWrappers::SparseMatrix &pressure_mass_,
                    const TrilinosWrappers::SparseMatrix &B_)
    {
      velocity_stiffness = &velocity_stiffness_;
      pressure_mass      = &pressure_mass_;
      B                  = &B_;

      preconditioner_velocity.initialize(velocity_stiffness_);
      preconditioner_pressure.initialize(pressure_mass_);
    }

    void vmult(TrilinosWrappers::MPI::BlockVector       &dst,
               const TrilinosWrappers::MPI::BlockVector &src) const
    {
      // 1) Solve velocity block
      {
        SolverControl                           solver_control_velocity(10000,
                                               1e-2 * src.block(0).l2_norm());
        SolverCG<TrilinosWrappers::MPI::Vector> solver_cg_velocity(solver_control_velocity);
        solver_cg_velocity.solve(*velocity_stiffness,
                                 dst.block(0),
                                 src.block(0),
                                 preconditioner_velocity);
      }
      // 2) Solve pressure block
      {
        tmp.reinit(src.block(1));
        B->vmult(tmp, dst.block(0));
        tmp.sadd(-1.0, src.block(1));

        SolverControl                           solver_control_pressure(10000,
                                               1e-2 * src.block(1).l2_norm());
        SolverCG<TrilinosWrappers::MPI::Vector> solver_cg_pressure(solver_control_pressure);
        solver_cg_pressure.solve(*pressure_mass,
                                 dst.block(1),
                                 tmp,
                                 preconditioner_pressure);
      }
    }

  protected:
    const TrilinosWrappers::SparseMatrix *velocity_stiffness = nullptr;
    const TrilinosWrappers::SparseMatrix *pressure_mass      = nullptr;
    const TrilinosWrappers::SparseMatrix *B                  = nullptr;

    TrilinosWrappers::PreconditionILU preconditioner_velocity;
    TrilinosWrappers::PreconditionILU preconditioner_pressure;

    mutable TrilinosWrappers::MPI::Vector tmp;
  };

protected:
  // ------------------------------------------------
  // Base class data members
  // ------------------------------------------------

  // Mesh and polynomial degrees
  std::string  mesh_file_name;
  unsigned int degree_velocity;
  unsigned int degree_pressure;

  // Problem constants
  const double H;      // Height of the Channel
  const double D;      // Diameter of the Cylinder
  const double uMax;   // Maximum Inflow Velocity
  const double uMean;  // Mean Inflow Velocity
  const double Re;     // Reynolds number
  const double p_out;  // Pressure at the outlet
  const double nu;     // Viscosity


  // MPI
  const unsigned int mpi_size;
  const unsigned int mpi_rank;
  ConditionalOStream pcout;

  // Triangulation and DoF
  parallel::fullydistributed::Triangulation<dim> mesh;
  DoFHandler<dim> dof_handler;

  // Finite element & quadrature
  std::unique_ptr<FiniteElement<dim>>   fe;
  std::unique_ptr<Quadrature<dim>>      quadrature;
  std::unique_ptr<Quadrature<dim - 1>>  quadrature_face;

  // Owned & relevant DoFs
  IndexSet               locally_owned_dofs;
  std::vector<IndexSet>  block_owned_dofs;
  IndexSet               locally_relevant_dofs;
  std::vector<IndexSet>  block_relevant_dofs;

  // Matrices & vectors
  TrilinosWrappers::BlockSparseMatrix system_matrix;
  TrilinosWrappers::BlockSparseMatrix pressure_mass;
  TrilinosWrappers::MPI::BlockVector  system_rhs;
  TrilinosWrappers::MPI::BlockVector  solution_owned;
  TrilinosWrappers::MPI::BlockVector  solution;


  // Forcing Term
  class ForcingTerm : public Function<dim>
  {
  public:
    void vector_value(const Point<dim> &, Vector<double> &values) const override
    {
      for (unsigned int i = 0; i < dim; ++i)
        values[i] = 0.0;
    }
    double value(const Point<dim> &, const unsigned int = 0) const override
    {
      return 0.0;
    }
  };

  // Inlet Velocity
  class InletVelocity : public Function<dim>
  {
  public:
    InletVelocity(const double H_in, const double uMax_in)
      : Function<dim>(dim + 1)
      , H(H_in)
      , uMax(uMax_in)
    {}

    void vector_value(const Point<dim> &p, Vector<double> &values) const override
    {
      if constexpr (dim == 2)
      {
        // 2D inlet velocity: 4 U_m y (H - y) / H^2
        values[0] = 4.0 * uMax * p[1] * (H - p[1]) / (H * H);
        for (unsigned int i = 1; i < dim + 1; ++i)
          values[i] = 0.0;
      }
      else if constexpr (dim == 3)
      {
        // 3D inlet velocity: 16 U_m y z (H - y)(H - z) / H^4 for U(0,y,z)
        values[0] = 16.0 * uMax * p[1] * p[2] * (H - p[1]) * (H - p[2]) / (H * H * H * H);
        for (unsigned int i = 1; i < dim + 1; ++i)
          values[i] = 0.0;
      }
    }

    double value(const Point<dim> &p, const unsigned int comp = 0) const override
    {
      if (comp == 0)
      {
        if constexpr (dim == 2)
          return 4.0 * uMax * p[1] * (H - p[1]) / (H * H);
        else if constexpr (dim == 3)
          return 16.0 * uMax * p[1] * p[2] * (H - p[1]) * (H - p[2]) / (H * H * H * H);
      }
      return 0.0;
    }

  private:
    const double H;
    const double uMax;
  };

  ForcingTerm   forcing_term;
  InletVelocity inlet_velocity;
};

// ----------------------------------------------------------------------------
// Derived class: Stokes<dim>
// ----------------------------------------------------------------------------
template <int dim>
class Stokes : public SteadyNavierStokes<dim>
{
public:
  Stokes(const std::string &mesh_file_name_in,
         unsigned int       degree_velocity_in,
         unsigned int       degree_pressure_in,
         double             Re_in)
    : SteadyNavierStokes<dim>(mesh_file_name_in,
                              degree_velocity_in,
                              degree_pressure_in,
                              Re_in)
  {
  }

  // Override the base workflow methods
  void setup() override;
  void assemble() override;
  void solve() override;
  void output() override;
  std::string get_output_directory() override;

  TrilinosWrappers::MPI::BlockVector get_solution() const
  {
    return this->solution;
  }
};

// ----------------------------------------------------------------------------
// Derived class: IncrementalStokes<dim>
// ----------------------------------------------------------------------------
template <int dim>
class IncrementalStokes : public SteadyNavierStokes<dim>
{
public:
  /**
   * Reuse the same parameters from a given Stokes object.
   * We call the SteadyNavierStokes base constructor with
   * the public getters from stokes_obj, then copy its mesh.
   */
  IncrementalStokes(const Stokes<dim> &stokes_obj)
    : SteadyNavierStokes<dim>(stokes_obj.get_mesh_file_name(),
                              stokes_obj.get_degree_velocity(),
                              stokes_obj.get_degree_pressure(),
                              stokes_obj.get_Re())
    , u_k(0)
    , p_k(dim)
  {
    // Copy the fully-distributed Triangulation
    this->mesh.copy_triangulation(stokes_obj.get_mesh());

    // Compute uMean and scaling_factor based on dimension
    if constexpr(dim == 2)
    {
      // Umean = (2/3) * U(0, H/2)
      double uMean = this->uMean;
      double D = this->D;
      
      // scaling_factor = 2 / (mean^2 * D)
      scaling_factor = 2.0 / (uMean * uMean * D);
      
      this->pcout << "2D Scaling Factor: " << scaling_factor << std::endl;
    }
    else if constexpr(dim == 3)
    {
      // Ustar = (4/9) * U(0, H/2, H/2)
      double uMean = this->uMean;
      double D = this->D;
      double H = this->H;
      
      // scaling_factor = 2 / (Umean^2 * D * H)
      scaling_factor = 2.0 / (uMean * uMean * D * H);
      
      this->pcout << "3D Scaling Factor: " << scaling_factor << std::endl;
    }
    else
    {
      static_assert(dim == 2 || dim == 3, "Dimension must be 2 or 3.");
    }
  }

  // Override the base workflow methods
  void setup() override;
  void assemble() override;
  void solve() override;
  void output() override;
  std::string get_output_directory() override;


  /**
   * Provide an initial condition for the iterative scheme.
   */
  void set_initial_conditions(const TrilinosWrappers::MPI::BlockVector solution_stokes_)
  {
    solution_old.reinit(solution_stokes_);
    solution_old = solution_stokes_;
  }

  /**
   * Post-processing function (lift/drag, etc.).
   */
  void compute_lift_drag();

protected:
  unsigned int       iter       = 0;
  const unsigned int maxIter    = 10;
  const double       update_tol = 1e-7;

  // Extractors for velocity & pressure
  FEValuesExtractors::Vector u_k;
  FEValuesExtractors::Scalar p_k;

  // Constraints
  AffineConstraints<double> constraints;

  // For iterative scheme
  TrilinosWrappers::MPI::BlockVector solution_old;
  TrilinosWrappers::MPI::BlockVector new_res;

  // Scaling factor for lift/drag computation
  double scaling_factor;

  // Example: Points of interest for any local postprocessing
  std::vector<Point<dim>> points_of_interest;
};


#endif // STEADYNAVIERSTOKES_HPP