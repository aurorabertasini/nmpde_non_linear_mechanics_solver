#ifndef LINARDO_HPP
#define LINARDO_HPP

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

// Base class for the problem.
class Linardo
{
public :
    Linardo(const std::string &mesh_file_name_, const double &re_)
    : mesh_file_name(mesh_file_name_)
    , mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)) 
    , mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
    , pcout(std::cout, mpi_rank == 0)
    , mesh(MPI_COMM_WORLD)
    , reynolds_number(re_)
    {
        this->nu = ( 2. * uM * 3. * cylinder_radius )  / reynolds_number;
    }

    virtual ~Linardo() = default;

    virtual void setup();
    virtual void assemble();
    virtual void solve();
    virtual void output();

    


    class PreconditionIdentity
    {
    public:
        // Application of the preconditioner: we just copy the input vector (src)
        // into the output vector (dst).
        void
        vmult(TrilinosWrappers::MPI::BlockVector       &dst,
            const TrilinosWrappers::MPI::BlockVector &src) const
        {
        dst = src;
        }

    protected:
    };

    class PreconditionBlockDiagonal
    {
    public:
        // Initialize the preconditioner, given the velocity stiffness matrix, the
        // pressure mass matrix.
        void
        initialize(const TrilinosWrappers::SparseMatrix &velocity_stiffness_,
                const TrilinosWrappers::SparseMatrix &pressure_mass_)
        {
        velocity_stiffness = &velocity_stiffness_;
        pressure_mass      = &pressure_mass_;

        preconditioner_velocity.initialize(velocity_stiffness_);
        preconditioner_pressure.initialize(pressure_mass_);
        }

        // Application of the preconditioner.
        void
        vmult(TrilinosWrappers::MPI::BlockVector       &dst,
            const TrilinosWrappers::MPI::BlockVector &src) const
        {
        SolverControl       solver_control_velocity(100000, 1e-2 * src.block(0).l2_norm());
        SolverCG<TrilinosWrappers::MPI::Vector> solver_cg_velocity(solver_control_velocity);
        solver_cg_velocity.solve(*velocity_stiffness,
                                dst.block(0),
                                src.block(0),
                                preconditioner_velocity);

        SolverControl                           solver_control_pressure(1000,
                                                1e-2 * src.block(1).l2_norm());
        SolverCG<TrilinosWrappers::MPI::Vector> solver_cg_pressure(
            solver_control_pressure);
        solver_cg_pressure.solve(*pressure_mass,
                                dst.block(1),
                                src.block(1),
                                preconditioner_pressure);
        }


    protected:
        // Velocity stiffness matrix.
        const TrilinosWrappers::SparseMatrix *velocity_stiffness;

        // Preconditioner used for the velocity block.
        TrilinosWrappers::PreconditionILU preconditioner_velocity;

        // Pressure mass matrix.
        const TrilinosWrappers::SparseMatrix *pressure_mass;

        // Preconditioner used for the pressure block.
        TrilinosWrappers::PreconditionILU preconditioner_pressure;
    };


    class PreconditionBlockTriangularStokes
    {
    public:
        // Initialize the preconditioner, given the velocity stiffness matrix, the
        // pressure mass matrix.
        void
        initialize(const TrilinosWrappers::SparseMatrix &velocity_stiffness_,
                const TrilinosWrappers::SparseMatrix &pressure_mass_,
                const TrilinosWrappers::SparseMatrix &B_)
        {
        velocity_stiffness = &velocity_stiffness_;
        pressure_mass      = &pressure_mass_;
        B                  = &B_;

        preconditioner_velocity.initialize(velocity_stiffness_);
        preconditioner_pressure.initialize(pressure_mass_);
        }

        // Application of the preconditioner.
        void
        vmult(TrilinosWrappers::MPI::BlockVector       &dst,
            const TrilinosWrappers::MPI::BlockVector &src) const
        {
        SolverControl solver_control_velocity(10000, 1e-2 * src.block(0).l2_norm());
        SolverCG<TrilinosWrappers::MPI::Vector> solver_cg_velocity(
            solver_control_velocity);
        solver_cg_velocity.solve(*velocity_stiffness,
                                dst.block(0),
                                src.block(0),
                                preconditioner_velocity);

        tmp.reinit(src.block(1));
        B->vmult(tmp, dst.block(0));
        tmp.sadd(-1.0, src.block(1));

        SolverControl   solver_control_pressure(10000,1e-2 * src.block(1).l2_norm());
        SolverCG<TrilinosWrappers::MPI::Vector> solver_cg_pressure(
            solver_control_pressure);
        solver_cg_pressure.solve(*pressure_mass,
                                dst.block(1),
                                tmp,
                                preconditioner_pressure);
        }



    protected:
        // Velocity stiffness matrix.
        const TrilinosWrappers::SparseMatrix *velocity_stiffness;

        // Preconditioner used for the velocity block.
        TrilinosWrappers::PreconditionILU preconditioner_velocity;

        // Pressure mass matrix.
        const TrilinosWrappers::SparseMatrix *pressure_mass;

        // Preconditioner used for the pressure block.
        TrilinosWrappers::PreconditionILU preconditioner_pressure;

        // B matrix.
        const TrilinosWrappers::SparseMatrix *B;

        // Temporary vector.
        mutable TrilinosWrappers::MPI::Vector tmp;
    };


    protected:
        const std::string mesh_file_name;
        static constexpr unsigned int dim = 2;
        const unsigned int mpi_size; // Number of MPI processes.
        const unsigned int mpi_rank; // Rank of this MPI process.
        ConditionalOStream pcout; // Parallel output stream.

        double nu; // Kinematic viscosity [m2/s].
        
        double uM = 0.3; // Maximum velocity [m/s].

        double cylinder_radius = 0.1; // Radius of the cylinder [m].

        const double reynolds_number; // Reynolds number.

        parallel::fullydistributed::Triangulation<dim> mesh; // Distributed mesh.
        std::unique_ptr<FiniteElement<dim>> fe; // Finite element.
        std::unique_ptr<Quadrature<dim>> quadrature; // Quadrature formula.
        std::unique_ptr<Quadrature<dim - 1>> quadrature_face; // Face quadrature formula.
        DoFHandler<dim> dof_handler; // DoF handler.
        IndexSet locally_owned_dofs;  // Locally owned DoFs.
        std::vector<IndexSet> block_owned_dofs; // Locally owned DoFs for each block.
        IndexSet locally_relevant_dofs; /// Locally relevant DoFs.
        std::vector<IndexSet> block_relevant_dofs; // Locally relevant DoFs for each block.

};

#endif

