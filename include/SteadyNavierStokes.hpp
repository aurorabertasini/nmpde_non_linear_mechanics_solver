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
#include <iostream>
#include <mpi.h>
#include <filesystem>

using namespace dealii;

/**
 * Base class: SteadyNavierStokes
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
                     const unsigned int degree_pressure_in)
    : mesh_file_name(mesh_file_name_in)
    , degree_velocity(degree_velocity_in)
    , degree_pressure(degree_pressure_in)   
    , nu(1.0)  
    , mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD))
    , mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
    , pcout(std::cout, mpi_rank == 0)
    , mesh(MPI_COMM_WORLD)
    , dof_handler(mesh)
    , forcing_term()
    , neumann_function(nu)
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
  void run_full_problem_pipeline(double& errorL2, double& errorH1);

  // -----------------------------------------------------------
  // Virtual methods for the typical solve steps
  // -----------------------------------------------------------
  virtual void setup();
  virtual void assemble();
  virtual void solve();
  virtual void output();
  double compute_error(const VectorTools::NormType &norm_type, bool velocity);
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


  // Provide access to the mesh
  const parallel::fullydistributed::Triangulation<dim> & get_mesh() const
  {
    return mesh;
  }

  // -----------------------------------------------------------
  // Custom Preconditioners
  // -----------------------------------------------------------

  // Trivial Preconditioner
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

protected:
  // ------------------------------------------------
  // Base class data members
  // ------------------------------------------------

  // Mesh and polynomial degrees
  std::string  mesh_file_name;
  unsigned int degree_velocity;
  unsigned int degree_pressure;


  // Problem constants
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

    // Class for the exact solution, containing both velocity and pressure.
  class ExactSolution : public Function<dim> {
   public:
    // Constructor.
    ExactSolution()
        : Function<dim>(dim + 1),
          exact_velocity(),
          exact_pressure() {}

    // When defining vector-valued functions, we need to define the value
    // function, which returns the value of the function at a given point and
    // component...
    virtual double value(const Point<dim> &p,
                         const unsigned int component) const override
    {
        if (component < dim) {
        return exact_velocity.value(p, component);
      } else if (component == dim) {
        return exact_pressure.value(p);
      } else {
        return 0.0;
      }
    }

    // ... and the vector_value function, which returns the value of the
    // function at a given point for all components.
    virtual void vector_value(const Point<dim> &p,
                              Vector<double> &values) const override
    {

      for (unsigned int i = 0; i < dim + 1; i++) {
        values[i] = value(p, i);
      }


    }

    // This is a function object, which defines the exact velocity. Since the
    // problem's exact solution is known, we can define it as a function object
    // and use it to compute the error of our numerical solution. To be able to
    // compute the H1 norm of the error, the exact gradient is computed as well.
    // This function returns 4 values, despite the fact that the last one is
    // empty, for compatibility reasons.
    class ExactVelocity : public Function<dim> {
     public:
      // Constructor.
      ExactVelocity() : Function<dim>(dim + 1){}

      // Evaluation.
      virtual double value(const Point<dim> &p,
                           const unsigned int component) const override
      {
        if (component == 0) {
          return std::cos(M_PI * p[0]);
        } else if (component == 1) {
          return p[1] * M_PI * std::sin(M_PI * p[0]);
        } else 
        {
          return 0;
        }

      }
    
    virtual void vector_value(const Point<dim> &p,
                                Vector<double> &values) const override
      {
        for (unsigned int i = 0; i < dim + 1; i++) {
        values[i] = value(p, i);
        }
      }
    
      virtual Tensor<1, dim> gradient(
          const Point<dim> &p, const unsigned int component) const override
      {
        Tensor<1, dim> result;
        if (component == 0)
        {
          result[0] = - M_PI * std::sin(M_PI * p[0]);
          result[1] = 0;
        }
        else if (component == 1)
        {
          result[0] = M_PI * p[1] * M_PI * std::cos(M_PI * p[0]);
          result[1] = M_PI * std::sin(M_PI * p[0]);
        }
        else
        {
          result[0] = 0;
          result[1] = 0;
        }
      return result;
      }

      virtual void vector_gradient(
          const Point<dim> &p,
          std::vector<Tensor<1, dim>> &values) const override
      {
        for (unsigned int i = 0; i < dim + 1; i++)
          values[i] = gradient(p, i);
      }

     private:
    };

    // Same as above, for the pressure. This is a scalar function since there is
    // no need to return 4 values due to the way it is used.
    class ExactPressure : public Function<dim> {
     public:
      // Constructor.
      ExactPressure() : Function<dim>(1){}

      // Evaluation.
      virtual double value(const Point<dim> &p,
                           const unsigned int /*component*/ = 0) const override
      {
        return p[0] * p[0] * p[1] * p[1]; 
      }
      virtual void vector_value(const Point<dim> &p,
                                Vector<double> &values) const override
      {
        values[0] = value(p);
      }

     private:

    };

   private:

   public:
    // This is a function object, which defines the exact velocity. Since we
    // need to set the time, it's defined as mutable.
    ExactVelocity exact_velocity;

    // This is a function object, which defines the exact pressure. Since we
    // need to set the time, it's defined as mutable.
    ExactPressure exact_pressure;
  };





  // Forcing Term
  class ForcingTerm : public Function<dim>
  {
  public:
  double value(const Point<dim> &p, const unsigned int component = 0) const override
  {
    double x = p[0];
    double y = p[1];

    double f_x = 2 * x * y * y - (M_PI * sin(2 * M_PI * x)) / 2 + (M_PI * M_PI * cos(M_PI * x)) / 1;
    double f_y = (y * (2 * x * x + M_PI * M_PI * M_PI * sin(M_PI * x) + 1 * M_PI * M_PI)) / 1;

    if (component == 0)
      return f_x;
    else if (component == 1)
      return f_y;
    else 
      return 0.0;
  }


    void vector_value(const Point<dim> &p, Vector<double> &values) const override
    {
      for (unsigned int i = 0; i < dim + 1; ++i)
        values[i] = value(p,i);
    }
    
  };



  // Class for the Neumann function, needed to impose the boundary conditions.
  class NeumannFunction : public Function<dim> {
   public:
    // Constructor.
    NeumannFunction(double nu_)
        : Function<dim>(dim + 1), nu(nu_), exact_solution() {}

    // Evaluation.
    virtual double value(
      const Point<dim> &p, const unsigned int component) const 
  {
    // Calculate the Neumann function.
    // This result was obtained by setting the normal vector to -i.
      if (component == 0) {
      Tensor<1, dim> velocity_gradient =
          exact_solution.exact_velocity.gradient(p, component);
      return -nu * velocity_gradient[0] + exact_solution.exact_pressure.value(p);
    } else if (component == 1) {
      Tensor<1, dim> velocity_gradient =
          exact_solution.exact_velocity.gradient(p, component);
      return -nu * velocity_gradient[0];
    } else {
      return 0.0;
    } 
  }

    virtual void vector_value(const Point<dim> &p,
                              Vector<double> &values) const override
    {
    for (unsigned int i = 0; i < dim; i++) {
      values[i] = value(p, i);
    }
    values[dim] = 0.0;
   }

   private:
    const double nu;

   public:
    // This is a function object, which defines the exact solution. Since we
    // need to set the time, it's defined as mutable.
    ExactSolution exact_solution;
  };

  ForcingTerm   forcing_term;
  ExactSolution exact_solution;
  NeumannFunction neumann_function;
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
         unsigned int       degree_pressure_in)
    : SteadyNavierStokes<dim>(mesh_file_name_in,
                              degree_velocity_in,
                              degree_pressure_in)
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
                              stokes_obj.get_degree_pressure())
    , u_k(0)
    , p_k(dim)
  {
    // Copy the fully-distributed Triangulation
    this->mesh.copy_triangulation(stokes_obj.get_mesh());
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



protected:
  const unsigned int maxIter    = 100;
  const double       update_tol = 6e-6;

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