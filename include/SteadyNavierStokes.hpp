#ifndef STEADYNAVIERSTOKES_HPP
#define STEADYNAVIERSTOKES_HPP

#include "../include/includes_file.hpp"

using namespace dealii;

// -----------------------------------------------------------
// Base class: SteadyNavierStokes
// -----------------------------------------------------------
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
    , Re(Re_in)     // User input
    , H(0.41)       // Fixed by geometry
    , D(0.1)        // Fixed by geometry
    , uMax(dim == 2 ? 0.3 : 0.45) // Fixed by problem
    , uMean(dim == 2 ? 2./3. * uMax : 4./9. * uMax) // Fixed by problem
    , nu(uMean * D / Re)  // Fixed by problem
    , p_out(0.0)    // Fixed by problem
    , forcing_term()
    , inlet_velocity(H, uMax)
    , mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD))
    , mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
    , pcout(std::cout, mpi_rank == 0)
    , mesh(MPI_COMM_WORLD)
    , dof_handler(mesh)

  {
  }

  // Disallow copy construction for clarity.
  SteadyNavierStokes(const SteadyNavierStokes<dim> &) = delete;
  virtual ~SteadyNavierStokes() = default;

  /**
   * A single method that instantiates a Stokes solver,
   * runs it, then instantiates a NonLinearCorrection solver,
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


  // -----------------------------------------------------------
  // Problem Customization Functions
  // -----------------------------------------------------------

  // Forcing Term
  /*
  *  A forcing term that is zero everywhere for flow past cylinder test case
  */
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
  /*
  *  Inlet velocity profile for flow past cylinder test case
  *  2D: U = 4 U_m y (H - y) / H^2, V = 0
  *  3D: U = 16 U_m y z (H - y)(H - z) / H^4, V = W = 0
  */
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
        values[0] = 4.0 * uMax * p[1] * (H - p[1]) / (H * H);
        for (unsigned int i = 1; i < dim + 1; ++i)
          values[i] = 0.0;
      }
      else if constexpr (dim == 3)
      {
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

protected:

  // ------------------------------------------------
  // Base class data members
  // ------------------------------------------------

  // Constructor Input Parameters
  std::string  mesh_file_name;              // Mesh file name
  const unsigned int degree_velocity;       // Velocity polynomial degree
  const unsigned int degree_pressure;       // Pressure polynomial degree
  const double Re;                          // Reynolds number

  // Geometrical & Physical Values
  const double H;                           // Height of the channel
  const double D;                           // Diameter of the obstacle
  const double uMax;                        // Maximum inflow velocity
  const double uMean;                       // Mean inflow velocity
  const double nu;                          // Viscosity
  const double p_out;                       // Outlet Neumann BC value

  // Post processing data
  double lift;
  double drag;
  double deltaP;

  // Problem-specific objects
  ForcingTerm   forcing_term;               // Forcing term
  InletVelocity inlet_velocity;             // Inlet velocity

  // MPI tools
  const unsigned int mpi_size;              // Number of MPI processes
  const unsigned int mpi_rank;              // Rank of the process   
  ConditionalOStream pcout;                 // Conditional output stream

  // FEM objects
  parallel::fullydistributed::Triangulation<dim> mesh;       // Parallel mesh
  DoFHandler<dim> dof_handler;                               // DoF handler      
  std::unique_ptr<FiniteElement<dim>>   fe;                  // Finite element handler
  std::unique_ptr<Quadrature<dim>>      quadrature;          // Domain quadrature 
  std::unique_ptr<Quadrature<dim - 1>>  quadrature_face;     // Face quadrature
  IndexSet               locally_owned_dofs;       // Locally owned DoFs
  std::vector<IndexSet>  block_owned_dofs;         // Block-wise owned DoFs
  IndexSet               locally_relevant_dofs;    // Locally relevant DoFs
  std::vector<IndexSet>  block_relevant_dofs;      // Block-wise relevant DoFs

  // Matrices & vectors
  TrilinosWrappers::BlockSparseMatrix system_matrix;     // System LHS matrix
  TrilinosWrappers::BlockSparseMatrix pressure_mass;     // Pressure mass matrix
  TrilinosWrappers::MPI::BlockVector  system_rhs;        // System RHS vector
  TrilinosWrappers::MPI::BlockVector  solution_owned;    // Solution vector (owned)
  TrilinosWrappers::MPI::BlockVector  solution;          // Solution vector (complete)
};


// ----------------------------------------------------------------------------
// Derived class: Stokes<dim>
// ----------------------------------------------------------------------------
/**
  * A derived class that solves the Stokes problem to provide
  * a proper initial solution for the Newton Problem, 
  * solved through the NonLinearCorrection class.   
  */
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
// Derived class: NonLinearCorrection
// ----------------------------------------------------------------------------

template <int dim>
class NonLinearCorrection : public SteadyNavierStokes<dim>
{
public:
  /**
   * Reuse the same parameters from a given Stokes object.
   * We call the SteadyNavierStokes base constructor with
   * the public getters from stokes_obj, then obtain its mesh.
   */
  NonLinearCorrection(const Stokes<dim> &stokes_obj)
    : SteadyNavierStokes<dim>(stokes_obj.get_mesh_file_name(),
                              stokes_obj.get_degree_velocity(),
                              stokes_obj.get_degree_pressure(),
                              stokes_obj.get_Re())
    , u_k(0)
    , p_k(dim)
  {
    // Copy the fully-distributed Triangulation
    this->mesh.copy_triangulation(stokes_obj.get_mesh());

    // Compute scaling_factor for lift and drag computation, based on dimension
    /*
    *  Scaling factor for flow past cylinder test case
    *  2D: scaling_factor = 2 / (Umean^2 * D)
    *  3D: scaling_factor = 2 / (Umean^2 * D * H)
    */
    double uMean = this->uMean;
    double D = this->D;
    if constexpr(dim == 2)
    {
      scaling_factor = 2.0 / (uMean * uMean * D);
    }
    else if constexpr(dim == 3)
    {
      double H = this->H;
      scaling_factor = 2.0 / (uMean * uMean * D * H);
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
   * Provide an initial condition for the iterative scheme
   */
  void set_initial_conditions(const TrilinosWrappers::MPI::BlockVector solution_stokes_)
  {
    solution_old.reinit(solution_stokes_);
    solution_old = solution_stokes_;
  }

  /**
   *  Compute lift, drag and DeltaP post processing function
   */
  void compute_lift_drag(double& lift, double& drag, double& delta_P);

protected:
  unsigned int       iter       = 0;      // Newton iterations counter
  const unsigned int maxIter    = 20;     // Maximum iterations
  const double       tolerance  = 1e-7;   // Update tolerance

  // Extractors and constraints
  FEValuesExtractors::Vector u_k;          // Velocity extractor
  FEValuesExtractors::Scalar p_k;          // Pressure extractor
  AffineConstraints<double> constraints;   // Constraints

  // For iterative scheme
  TrilinosWrappers::MPI::BlockVector solution_old;  // Old solution
  TrilinosWrappers::MPI::BlockVector new_res;       // Residual at current iteration

  // Post-processing data 
  double scaling_factor;                        // Scaling factor for lift and drag
  std::vector<Point<dim>> points_of_interest;   // Points of interest for pressure difference
};


#endif // STEADYNAVIERSTOKES_HPP