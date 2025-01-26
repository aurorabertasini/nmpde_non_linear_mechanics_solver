#ifndef NAVIERSTOKES_HPP
#define NAVIERSTOKES_HPP

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_values_extractors.h>
#include <deal.II/fe/mapping_fe.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria.h>

#include <deal.II/grid/grid_in.h>

#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

// ---------------------------------------------------------------------

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_values_extractors.h>
#include <deal.II/fe/mapping_fe.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/grid/grid_in.h>

#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/trilinos_block_sparse_matrix.h>
#include <deal.II/lac/trilinos_parallel_block_vector.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

// ---------------------------------------------------------------------

#include <filesystem>
#include <fstream>
#include <iostream>

using namespace dealii;

// to write function in space use p[0], p[1], p[2], get_time() for x, y, z, t respectively
// to write exponential function use pow(base, exp), for example pow(M_E, 2*p[0]) for exp(2x)
// to write sin function use std::sin, for example std::sin(M_PI*p[0]) for sin(pi*x)

class MonolithicNavierStokes
{
public:
    // Physical dimension (1D, 2D, 3D)
    static constexpr unsigned int dim = 2;

    // 2. Function for the transport coefficient.
    class Function_t : public Function<dim>
    {
    public:
        virtual void
        vector_value(const Point<dim> & /*p*/,
                     Vector<double> &values) const override
        {
            values[0] = 0.0;
            values[1] = 0.0;
        }
        virtual double
        value(const Point<dim> & /*p*/,
              const unsigned int component = 0) const override
        {
            if (component == 0)
                return 0.0;
            else // if (component == 1)
                return 0.0;
        }
    };

    class ForcingTerm : public Function<dim>
    {
    public:
        virtual void
        vector_value(const Point<dim> & /*p*/,
                     Vector<double> &values) const override
        {
            for (unsigned int i = 0; i < dim; ++i)
                values[i] = 0.0;
        }

        virtual double
        value(const Point<dim> & /*p*/,
              const unsigned int /*component*/ = 0) const override
        {
            return 0.0;
        }

    protected:
    };

    class InletVelocity : public Function<dim>
    {
    public:
        InletVelocity(const double H)
            : Function<dim>(dim + 1)
        {
            this->H = H;
        }

        virtual void
        vector_value(const Point<dim> &p, Vector<double> &values) const override
        {
            values[0] = 4.0 * uM * p[1] * (H - p[1]) / (H * H);

            for (unsigned int i = 1; i < dim + 1; ++i)
                values[i] = 0.0;
        }

        virtual double
        value(const Point<dim> &p, const unsigned int component = 0) const override
        {
            if (component == 0)
                return 4.0 * uM * p[1] * (H - p[1]) / (H * H);
            else
                return 0.0;
        }

        double get_u_max() const
        {
            return uM;
        }

    protected:
        const double uM = 1.5;
        double H;
    };

    // function for hom dirichelet condition
    class Function_Dirichelet_hom : public Function<dim>
    {
    public:
        Function_Dirichelet_hom()
            : Function<dim>(dim + 1)
        {
        }

        virtual void
        vector_value(const Point<dim> &p, Vector<double> &values) const override
        {
            values[0] = 0.0;

            for (unsigned int i = 1; i < dim + 1; ++i)
                values[i] = 0.0;
        }

        virtual double
        value(const Point<dim> &p, const unsigned int component = 0) const override
        {
            if (component == 0)
                return 0.0;
            else
                return 0.0;
        }
    };

    // function for Neumann condition
    class Function_Neumann : public Function<dim>
    {
    public:
        Function_Neumann()
            : Function<dim>(dim + 1)
        {
        }

        virtual void
        vector_value(const Point<dim> &p, Vector<double> &values) const override
        {
            values[0] = 0.0;

            for (unsigned int i = 1; i < dim + 1; ++i)
                values[i] = 0.0;
        }

        virtual double
        value(const Point<dim> &p, const unsigned int component = 0) const override
        {
            if (component == 0)
                return 0.0;
            else
                return 0.0;
        }
    };

    // Function for the initial condition.
    class Function_u0 : public Function<dim>
    {
    public:
        virtual void
        vector_value(const Point<dim> &p, Vector<double> &values) const override
        {
            values[0] = 0.0;

            for (unsigned int i = 1; i < dim + 1; ++i)
                values[i] = 0.0;
        }

        virtual double
        value(const Point<dim> &p, const unsigned int component = 0) const override
        {
            if (component == 0)
                return 0.0;
            else
                return 0.0;
        }
    };

    // // Exact solution.
    //   class ExactSolution : public Function<dim>
    //   {
    //   public:
    //     virtual double
    //     value(const Point<dim> &p,
    //           const unsigned int /*component*/ = 0) const override
    //     {
    //       return 0.0;
    //     }

    //     virtual Tensor<1, dim>
    //     gradient(const Point<dim> &p,
    //             const unsigned int /*component*/ = 0) const override
    //     {
    //       Tensor<1, dim> result;

    //       // duex / dx
    //       result[0] = 0.0;

    //       // duex / dy
    //       result[1] = 0.0;

    //       return result;
    //     }
    //   };

    // Identity preconditioner.
    class PreconditionIdentity
    {
    public:
        // Application of the preconditioner: we just copy the input vector (src)
        // into the output vector (dst).
        void
        vmult(TrilinosWrappers::MPI::BlockVector &dst,
              const TrilinosWrappers::MPI::BlockVector &src) const
        {
            dst = src;
        }

    protected:
    };

    // Constructor. We provide the final time, time step Delta t and theta method
    // parameter as constructor arguments.
    MonolithicNavierStokes(
        const std::string &mesh_file_name_,
        const unsigned int &degree_velocity_,
        const unsigned int &degree_pressure_,
        const double &T_,
        const double &deltat_,
        const double &theta_,
        const double &re_)
        : mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD))
        , mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
        , pcout(std::cout, mpi_rank == 0)
        , mesh_file_name(mesh_file_name_)
        , degree_velocity(degree_velocity_)
        , degree_pressure(degree_pressure_)
        , T(T_)
        , deltat(deltat_)
        , theta(theta_)
        , inlet_velocity(H)
        , velocity(0)
        , pressure(dim)
        , mesh(MPI_COMM_WORLD)
        , reynolds_number(re_)
    {
        this->nu = (2. * inlet_velocity.get_u_max() * 3. * cylinder_radius) / reynolds_number;
    }

    // Initialization.
    void
    setup();

    // Solve the problem.
    void
    solve();

    // Compute the error.
    double
    compute_error(const VectorTools::NormType &norm_type);

    std::string
    get_output_directory();

protected:
    // Assemble the matrix without the non-linear term.
    void
    assemble_base_matrix();

    void
    add_convective_term();

    // Assemble the right-hand side of the problem.
    void
    assemble_rhs(const double &time);

    // Solve the problem for one time step.
    void
    solve_time_step();

    // Output.
    void
    output(const unsigned int &time_step);

    void
    set_initial_conditions(TrilinosWrappers::MPI::BlockVector solution_stokes_);

    void
    compute_lift_drag();

    // MPI parallel. /////////////////////////////////////////////////////////////

    // extractors for velocity and pressure.
    FEValuesExtractors::Vector velocity;

    FEValuesExtractors::Scalar pressure;

    // Number of MPI processes.
    const unsigned int mpi_size;

    // This MPI process.
    const unsigned int mpi_rank;

    // Affine constraints.
    AffineConstraints<double> constraints;

    // diffusion coefficient
    double nu;

    // Reynolds number.
    const double reynolds_number;

    // Height of the channel.
    const double H = 0.41;

    // Inlet velocity.
    InletVelocity inlet_velocity;

    // Cylinder radius.
    const double cylinder_radius = 0.1;

    // // Transport coefficient.
    // Function_t transport;

    // outlet pressure
    const double p_out = 0.0;

    // Forcing term.
    ForcingTerm forcing_term;

    // Function for neuman condition
    Function_Neumann neumann_condition;

    // Function for the initial condition
    Function_u0 initial_condition;

    // // Exact solution.
    // ExactSolution exact_solution;

    // Current time.
    double time;

    // Final time.
    const double T;

    double lift; 

    double drag;

    const double rho = 1.0;

    // Discretization. ///////////////////////////////////////////////////////////

    // Mesh file name.
    const std::string mesh_file_name;

    // Polynomial degree Velocity.
    const unsigned int degree_velocity;

    // Polynomial degree Pressure.
    const unsigned int degree_pressure;

    // Time step.
    const double deltat;

    // Theta parameter of the theta method.
    const double theta;

    // Mesh.
    parallel::fullydistributed::Triangulation<dim> mesh;

    // Finite element space.
    std::unique_ptr<FiniteElement<dim>> fe;

    // Quadrature formula.
    std::unique_ptr<Quadrature<dim>> quadrature;

    // Quadrature formula used on boundary lines.
    std::unique_ptr<Quadrature<dim - 1>> quadrature_face;

    // DoF handler.
    DoFHandler<dim> dof_handler;

    // DoFs owned by current process.
    IndexSet locally_owned_dofs;

    std::vector<IndexSet> block_owned_dofs; //!!!!!!!!!!!!!!!!!!

    // DoFs relevant to the current process (including ghost DoFs).
    IndexSet locally_relevant_dofs;

    std::vector<IndexSet> block_relevant_dofs; //!!!!!!!!!!!!!!!!!!

    // Mass matrix M / deltat.
    TrilinosWrappers::BlockSparseMatrix mass_matrix;

    // Global matrix.
    TrilinosWrappers::BlockSparseMatrix system_matrix;

    // Non_linear term matrix.
    TrilinosWrappers::BlockSparseMatrix convective_matrix;

    // boolean value to choose to reconstruct all the system matrix ad each time step
    bool RECONSTRUCT_SYSTEM_MATRIX = true;

    // Stiffness matrix K.
    TrilinosWrappers::BlockSparseMatrix stiffness_matrix;

    // // Matrix on the left-hand side (M / deltat + theta A).
    // TrilinosWrappers::BlockSparseMatrix lhs_matrix;

    // Matrix on the right-hand side (M / deltat - (1 - theta) A).
    TrilinosWrappers::BlockSparseMatrix rhs_matrix;

    // Right-hand side vector in the linear system.
    TrilinosWrappers::MPI::BlockVector system_rhs;

    // System solution (without ghost elements).
    TrilinosWrappers::MPI::BlockVector solution_owned;

    // System solution (including ghost elements).
    TrilinosWrappers::MPI::BlockVector solution;

    ConditionalOStream pcout; // Parallel output stream.

    std::vector<double> vec_drag;
    std::vector<double> vec_lift;
    std::vector<double> vec_drag_coeff;
    std::vector<double> vec_lift_coeff;
};

#endif