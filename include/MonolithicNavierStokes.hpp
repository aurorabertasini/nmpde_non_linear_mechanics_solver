#ifndef MONOLITHICNAVIERSTOKES_HPP
#define MONOLITHICNAVIERSTOKES_HPP

#include "includes_file.hpp"
using namespace dealii;

// ==================================================================
// Class: MonolithicNavierStokes
//
// Description:
//   This class solves the incompressible Navier-Stokes equations
//   using a monolithic approach. The class is templated on the
//   dimensionality of the problem in order to handle 2D and 3D
//   problems. The class provides a variety of options for
//   preconditioners, including SIMPLE, aSIMPLE, and YOSIDA.
//
//  =================================================================

template <unsigned int dim>
class MonolithicNavierStokes
{
public:
    // ---------------------------------------------------------------
    // Class: InletVelocity
    //
    // Description:
    //   This class defines an inlet velocity function.
    //   It sets the maximum velocity value (uM) based on the
    //   dimensionality of the problem: 1.5 for 2D problems, 2.25 for 3D problems.
    //
    // Parameters:
    //   H - characteristic length of the domain.
    //   Um - maximum velocity value.
    // ---------------------------------------------------------------
    class InletVelocity : public Function<dim>
    {
    public:
        InletVelocity(const double H_)
            : Function<dim>(dim), H(H_)
        {
            if constexpr (dim == 2)
                this->uM = 1.5;
            else if constexpr (dim == 3)
                this->uM = 2.25;
        }

        virtual void
        vector_value(const Point<dim> &p, Vector<double> &values) const override
        {
            if constexpr (dim == 2)
                values[0] = 4.0 * uM * p[1] * (H - p[1]) / (H * H);
            else if constexpr (dim == 3)
                values[0] = 16.0 * uM * p[1] * (H - p[1]) * p[2] * (H - p[2]) / (H * H * H * H);
            for (unsigned int i = 1; i < dim; ++i)
                values[i] = 0.0;
        }

        virtual double
        value(const Point<dim> &p, const unsigned int component = 0) const override
        {
            if (component == 0)
            {
                if constexpr (dim == 2)
                    return 4.0 * uM * p[1] * (H - p[1]) / (H * H);
                else if constexpr (dim == 3)
                    return 16.0 * uM * p[1] * (H - p[1]) * p[2] * (H - p[2]) / (H * H * H * H);
            }
            else
                return 0.0;
        }

        double get_u_max() const
        {
            return uM;
        }

    protected:
        double uM;
        const double H;
    };

    /*****************************************************************/
    /*                 PRECONDITIONERS OPTIONS                       */
    /*****************************************************************/

    // ---------------------------------------------------------------
    // Class: BlockPrecondition
    //
    // Description:
    //   This class defines an abstract class for block preconditioners.
    //   It provides a virtual function to apply the preconditioner.
    // ---------------------------------------------------------------
    class BlockPrecondition
    {
    public:
        virtual ~BlockPrecondition() = default;
        virtual void vmult(TrilinosWrappers::MPI::BlockVector &dst,
                           const TrilinosWrappers::MPI::BlockVector &src) const = 0;

    protected:
        void initialize_inner_preconditioner(
            std::shared_ptr<TrilinosWrappers::PreconditionBase> &preconditioner,
            const TrilinosWrappers::SparseMatrix &matrix, bool use_ilu)
        {
            if (use_ilu)
            {
                std::shared_ptr<TrilinosWrappers::PreconditionILU> actual_preconditioner =
                    std::make_shared<TrilinosWrappers::PreconditionILU>();
                actual_preconditioner->initialize(matrix);
                preconditioner = actual_preconditioner;
            }
            else
            {
                std::shared_ptr<TrilinosWrappers::PreconditionAMG> actual_preconditioner =
                    std::make_shared<TrilinosWrappers::PreconditionAMG>();
                actual_preconditioner->initialize(matrix);
                preconditioner = actual_preconditioner;
            }
        }
    };

    // ---------------------------------------------------------------
    // Class: PreconditionSIMPLE
    //
    // Description:
    //   This class defines the SIMPLE preconditioner for solving block-structured
    //   systems of equations. It is designed to work with a system matrix partitioned
    //   as follows:
    //
    //       [ C   B^T ]
    //       [ B  -S   ]
    //
    //   where C is the (1,1)-block and B (and its transpose) appear in the
    //   off-diagonals. The preconditioner applies an approximate block factorization in
    //   two steps.
    //
    // ---------------------------------------------------------------
    class PreconditionSIMPLE : public BlockPrecondition
    {
    public:
        // Initialize the preconditioner.
        //
        // Parameters:
        //   C_matrix_     - Sparse matrix representing the top-left block (C).
        //   negB_matrix_  - Sparse matrix representing the negative of B (bottom-left).
        //   Bt_matrix_    - Sparse matrix representing the transpose of B (top-right).
        //   vec           - Block vector providing the structure for the first block.
        //   alpha_        - Damping parameter (in (0,1]) for final scaling.
        //   maxit_        - Maximum iterations for inner solvers.
        //   tol_          - Tolerance for convergence in inner solvers.
        //   use_ilu       - Boolean flag to enable ILU factorization in inner solvers.
        void initialize(const TrilinosWrappers::SparseMatrix &C_matrix_,
                        const TrilinosWrappers::SparseMatrix &negB_matrix_,
                        const TrilinosWrappers::SparseMatrix &Bt_matrix_,
                        const TrilinosWrappers::MPI::BlockVector &vec,
                        const double &alpha_, const unsigned int &maxit_,
                        const double &tol_, const bool &use_ilu)
        {
            // Save input parameters.
            alpha = alpha_;
            maxit = maxit_;
            tol = tol_;

            // Store pointers to the system matrices.
            C_matrix = &C_matrix_;
            negB_matrix = &negB_matrix_;
            Bt_matrix = &Bt_matrix_;

            // Compute and store the negative inverse of the diagonal entries of C.
            // The vector negDinv_vector will later be used for scaling operations.
            negDinv_vector.reinit(vec.block(0));
            for (unsigned int index : negDinv_vector.locally_owned_elements())
            {
                // Avoid division by zero by assuming diag_element(index) is nonzero.
                negDinv_vector[index] = -1.0 / C_matrix->diag_element(index);
            }

            // Build the auxiliary matrix S.
            // S is defined as S = B * (D^-1) * B^T,
            // where D^-1 is represented by the vector negDinv_vector (note the negative sign).
            // Here, mmult performs the matrix-matrix multiplication incorporating the scaling.
            negB_matrix->mmult(S_matrix, *Bt_matrix, negDinv_vector);

            // Initialize the inner preconditioners for both the C block and the Schur complement S.
            // These preconditioners (preconditioner_C for C and preconditioner_S for S) will be
            // used to solve the corresponding subsystems iteratively.
            this->initialize_inner_preconditioner(preconditioner_C, *C_matrix, use_ilu);
            this->initialize_inner_preconditioner(preconditioner_S, S_matrix, use_ilu);
        }

        // Apply the SIMPLE preconditioner.
        //
        // This method transforms a given block vector (src) by applying an approximate
        // inverse of the system matrix in two main steps:
        // 1. Solve the block lower triangular system:
        //      [ C    0 ] [ sol1_u ] = [ src_u ]
        //      [ B   -S ] [ sol1_p ]   [ src_p ]
        // 2. Solve the subsequent system to correct the intermediate solution:
        //      [ I    D^-1*B^T ] [ dst_u ] = [ sol1_u ]
        //      [ 0      alpha  ] [ dst_p ]   [ sol1_p ]
        //
        // Parameters:
        // - dst: The block vector where the preconditioned result is stored.
        // - src: The input block vector to which the preconditioner is applied.
        void vmult(TrilinosWrappers::MPI::BlockVector &dst,
                   const TrilinosWrappers::MPI::BlockVector &src) const
        {
            // Create a temporary block vector to store intermediate results.
            tmp.reinit(src);

            // =====================================================
            // Step 1: Solve the lower triangular system
            //         [ C   0 ]
            //         [ B  -S ] sol1 = src
            // =====================================================

            // Step 1.1: Solve for the velocity-like component (u-part):
            //         C * sol1_u = src_u
            // Here, we solve the linear system using GMRES with preconditioning.
            SolverControl solver_control_C(maxit, tol * src.block(0).l2_norm());
            SolverGMRES<TrilinosWrappers::MPI::Vector> solver_C(solver_control_C);
            solver_C.solve(*C_matrix, tmp.block(0), src.block(0), *preconditioner_C);

            // Step 1.2: Solve for the pressure-like component (p-part):
            //         S * sol1_p = B * sol1_u - src_p
            // First, compute B * sol1_u and subtract src_p.
            Bt_matrix->Tvmult(tmp.block(1), tmp.block(0));
            tmp.block(1) -= src.block(1);

            // Solve for sol1_p using GMRES with the corresponding preconditioner.
            SolverControl solver_control_S(maxit, tol * tmp.block(1).l2_norm());
            SolverGMRES<TrilinosWrappers::MPI::Vector> solver_S(solver_control_S);
            solver_S.solve(S_matrix, dst.block(1), tmp.block(1), *preconditioner_S);

            // =====================================================
            // Step 2: Solve the correction system
            //         [ I   D^-1*B^T ]
            //         [ 0     alpha  ] dst = sol1
            // =====================================================

            // Step 2.1: Correct the pressure-like component.
            //         Here, we scale sol1_p by 1/alpha.
            dst.block(1) /= alpha;

            // Step 2.2: Correct the velocity-like component.
            //         Compute D^-1*B^T*dst_p and subtract it from sol1_u.
            //         This involves:
            //         - Computing B^T*dst_p.
            //         - Scaling the result element-wise with negDinv_vector (which stores -1/diag(C)).
            //         - Adding the scaled vector to sol1_u.
            dst.block(0) = tmp.block(0);                  // Start with sol1_u.
            Bt_matrix->vmult(tmp.block(0), dst.block(1)); // Compute B^T*dst_p.
            tmp.block(0).scale(negDinv_vector);           // Multiply by D^-1 (with a negative sign).
            dst.block(0) += tmp.block(0);                 // Update the u-part.
        }

    private:
        double alpha;

        const TrilinosWrappers::SparseMatrix *C_matrix;

        const TrilinosWrappers::SparseMatrix *negB_matrix;

        const TrilinosWrappers::SparseMatrix *Bt_matrix;

        TrilinosWrappers::MPI::Vector negDinv_vector;

        TrilinosWrappers::SparseMatrix S_matrix;

        std::shared_ptr<TrilinosWrappers::PreconditionBase> preconditioner_C;

        std::shared_ptr<TrilinosWrappers::PreconditionBase> preconditioner_S;

        mutable TrilinosWrappers::MPI::BlockVector tmp;

        unsigned int maxit;

        double tol;
    };

    // ---------------------------------------------------------------
    // Class: PreconditionaSIMPLE
    //
    // Overview:
    //   This class implements an alternative SIMPLE preconditioner for solving
    //   block-structured systems. The system is assumed to have the form:
    //
    //         [ C    B^T ]
    //         [ B   ?    ]
    //
    //   Here, the method uses the diagonal of C to build two helper vectors:
    //     - D_vector: stores the diagonal entries of C,
    //     - Dinv_vector: stores the reciprocal of these entries.
    //   The auxiliary matrix is then formed as: negS_matrix = -B * Dinv_vector * B^T.
    //   The preconditioning action is achieved by a sequence of block operations
    //   that involve solving with C and with negS_matrix, scaling by D_vector,
    //   and applying corrections based on B and its transpose.
    // ---------------------------------------------------------------
    class PreconditionaSIMPLE : public BlockPrecondition
    {
    public:
        // Initializes the aSIMPLE preconditioner.
        //
        // This function:
        //  - Stores the provided matrices and parameters.
        //  - Extracts the diagonal of C (the top-left block) and computes its inverse.
        //  - Uses the inverse diagonal to build an approximation of the Schur complement,
        //    stored in negS_matrix.
        //  - Sets up the internal solvers for the C and negS_matrix subsystems.
        //
        // Parameters:
        //   C_matrix_    : Matrix representing the top-left block.
        //   negB_matrix_ : Matrix corresponding to -B (the bottom-left block).
        //   Bt_matrix_   : Matrix representing B^T (the top-right block).
        //   vec          : Block vector used to set up the layout of the diagonal.
        //   alpha_       : Damping parameter for later scaling (should be in (0,1]).
        //   maxit_       : Maximum iterations for the inner solvers.
        //   tol_         : Tolerance for the inner iterative solves.
        //   use_ilu      : Boolean flag indicating whether to use ILU preconditioning.
        void initialize(const TrilinosWrappers::SparseMatrix &C_matrix_,
                        const TrilinosWrappers::SparseMatrix &negB_matrix_,
                        const TrilinosWrappers::SparseMatrix &Bt_matrix_,
                        const TrilinosWrappers::MPI::BlockVector &vec,
                        const double &alpha_, const unsigned int &maxit_,
                        const double &tol_, const bool &use_ilu)
        {
            // Record the damping factor and solver parameters.
            alpha = alpha_;
            maxit = maxit_;
            tol = tol_;

            // Save pointers to the provided matrices.
            C_matrix = &C_matrix_;
            negB_matrix = &negB_matrix_;
            Bt_matrix = &Bt_matrix_;

            // Prepare vectors to store the diagonal of C and its reciprocal.
            // These will be used to weight the contributions from B and B^T.
            D_vector.reinit(vec.block(0));
            Dinv_vector.reinit(vec.block(0));
            for (unsigned int index : D_vector.locally_owned_elements())
            {
                const double value = C_matrix->diag_element(index);
                D_vector[index] = value;
                Dinv_vector[index] = 1.0 / value;
            }

            // Form the auxiliary matrix that approximates the Schur complement.
            // The operation computes: negS_matrix = -B * (Dinv) * B^T.
            negB_matrix->mmult(negS_matrix, *Bt_matrix, Dinv_vector);

            // Set up inner iterative solvers for the C block and the approximate
            // Schur complement (negS_matrix), possibly using ILU if indicated.
            this->initialize_inner_preconditioner(preconditioner_C, *C_matrix, use_ilu);
            this->initialize_inner_preconditioner(preconditioner_S, negS_matrix, use_ilu);
        }

        // Applies the aSIMPLE preconditioner to the input vector.
        //
        // The process is broken down into several operations:
        //
        // 1. Compute an initial approximation for the primary variable by solving
        //    C * x = src_primary.
        //
        // 2. Form an intermediate residual for the secondary variable by taking
        //    src_secondary and adding a contribution from the primary variable
        //    (via -B * x).
        //
        // 3. Solve for the secondary variable using the approximate Schur complement:
        //    negS_matrix * y = (intermediate residual).
        //
        // 4. Scale the primary component by the original diagonal of C.
        //
        // 5. Apply a correction to the primary component by subtracting B^T times the
        //    computed secondary variable.
        //
        // 6. Finalize the primary variable by re-scaling with the inverse of the diagonal.
        //
        // The result is written to dst.
        //
        // Parameters:
        //   dst - The output vector after preconditioning.
        //   src - The input vector to be preconditioned.
        void vmult(TrilinosWrappers::MPI::BlockVector &dst,
                   const TrilinosWrappers::MPI::BlockVector &src) const
        {
            // Prepare a temporary block vector to hold intermediate data.
            tmp.reinit(src);

            // --- Step 1 ---
            // Solve for the primary (first block) variable.
            // This computes an approximate inverse of C applied to the first part of src.
            SolverControl solver_control_C(maxit, tol * src.block(0).l2_norm());
            SolverGMRES<TrilinosWrappers::MPI::Vector> solver_C(solver_control_C);
            solver_C.solve(*C_matrix, dst.block(0), src.block(0), *preconditioner_C);

            // --- Step 2 ---
            // Copy the secondary part of src into a temporary container.
            // Then update this temporary vector by incorporating the effect of -B acting
            // on the primary variable obtained in Step 1.
            tmp.block(1) = src.block(1);
            negB_matrix->vmult_add(tmp.block(1), dst.block(0));

            // --- Step 3 ---
            // Solve the system with the approximate Schur complement.
            // This computes the secondary variable by inverting negS_matrix.
            SolverControl solver_control_S(maxit, tol * tmp.block(1).l2_norm());
            SolverGMRES<TrilinosWrappers::MPI::Vector> solver_S(solver_control_S);
            solver_S.solve(negS_matrix, dst.block(1), tmp.block(1), *preconditioner_S);

            // --- Step 4 ---
            // Scale the primary component by the original diagonal entries.
            // This reintroduces the proper weighting based on C's diagonal.
            dst.block(0).scale(D_vector);

            // --- Step 5 ---
            // Adjust the secondary component by applying the damping factor.
            dst.block(1) /= alpha;

            // --- Step 6 ---
            // Refine the primary component: subtract a correction term computed by
            // multiplying the secondary variable with B^T and then scaling with the inverse
            // of the diagonal entries.
            Bt_matrix->vmult(tmp.block(0), dst.block(1));
            dst.block(0) -= tmp.block(0);

            // --- Step 7 ---
            // Finalize the update of the primary component by scaling it with Dinv.
            dst.block(0).scale(Dinv_vector);
        }

    private:

        double alpha;

        const TrilinosWrappers::SparseMatrix *C_matrix;

        const TrilinosWrappers::SparseMatrix *negB_matrix;

        const TrilinosWrappers::SparseMatrix *Bt_matrix;

        TrilinosWrappers::MPI::Vector D_vector;

        TrilinosWrappers::MPI::Vector Dinv_vector;

        TrilinosWrappers::SparseMatrix negS_matrix;

        std::shared_ptr<TrilinosWrappers::PreconditionBase> preconditioner_C;

        std::shared_ptr<TrilinosWrappers::PreconditionBase> preconditioner_S;

        mutable TrilinosWrappers::MPI::BlockVector tmp;

        unsigned int maxit;

        double tol;
    };

    // Yosida preconditioner.
    class PreconditionYosida : public BlockPrecondition
    {
    public:
        // Initialize the preconditioner.
        void initialize(const TrilinosWrappers::SparseMatrix &C_matrix_,
                        const TrilinosWrappers::SparseMatrix &negB_matrix_,
                        const TrilinosWrappers::SparseMatrix &Bt_matrix_,
                        const TrilinosWrappers::SparseMatrix &M_dt_matrix_,
                        const TrilinosWrappers::MPI::BlockVector &vec,
                        const unsigned int &maxit_, const double &tol_,
                        const bool &use_ilu)
        {
            maxit = maxit_;
            tol = tol_;
            // Save a reference to the input matrices.
            C_matrix = &C_matrix_;
            negB_matrix = &negB_matrix_;
            Bt_matrix = &Bt_matrix_;

            // Save the inverse diagonal of M_dt.
            Dinv_vector.reinit(vec.block(0));
            for (unsigned int index : Dinv_vector.locally_owned_elements())
            {
                Dinv_vector[index] = 1.0 / M_dt_matrix_.diag_element(index);
            }

            // Create the matrix -S.
            negB_matrix->mmult(negS_matrix, *Bt_matrix, Dinv_vector);

            // Initialize the preconditioners.
            this->initialize_inner_preconditioner(preconditioner_C, *C_matrix, use_ilu);
            this->initialize_inner_preconditioner(preconditioner_S, negS_matrix, use_ilu);
        }

        // Application of the preconditioner.
        void vmult(TrilinosWrappers::MPI::BlockVector &dst,
                   const TrilinosWrappers::MPI::BlockVector &src) const
        {
            tmp.reinit(src);
            // Step 1: solve [C0; B -S]sol1 = src.
            // Step 1.1: solve C*sol1_u = src_u.
            tmp.block(0) = dst.block(0);
            SolverControl solver_control_C(maxit, tol * src.block(0).l2_norm());
            SolverGMRES<TrilinosWrappers::MPI::Vector> solver_C(solver_control_C);
            solver_C.solve(*C_matrix, tmp.block(0), src.block(0), *preconditioner_C);
            // Step 1.2: solve -S*sol1_p = -B*sol1_u + src_p.
            tmp.block(1) = src.block(1);
            negB_matrix->vmult_add(tmp.block(1), tmp.block(0));
            SolverControl solver_control_S(maxit, tol * tmp.block(1).l2_norm());
            SolverGMRES<TrilinosWrappers::MPI::Vector> solver_S(solver_control_S);
            solver_S.solve(negS_matrix, dst.block(1), tmp.block(1), *preconditioner_S);

            // Step 2: solve [I F^-1*B^T; 0 I]dst = sol1.
            tmp_2 = src.block(0);
            dst.block(0) = tmp.block(0);
            Bt_matrix->vmult(tmp.block(0), dst.block(1));
            SolverControl solver_control_C2(maxit, tol * tmp.block(0).l2_norm());
            SolverGMRES<TrilinosWrappers::MPI::Vector> solver_gmres_C2(solver_control_C);
            solver_gmres_C2.solve(*C_matrix, tmp_2, tmp.block(0), *preconditioner_C);
            dst.block(0) -= tmp_2;
        }

    private:
        const TrilinosWrappers::SparseMatrix *C_matrix;

        const TrilinosWrappers::SparseMatrix *negB_matrix;

        const TrilinosWrappers::SparseMatrix *Bt_matrix;

        TrilinosWrappers::MPI::Vector Dinv_vector;

        TrilinosWrappers::SparseMatrix negS_matrix;

        std::shared_ptr<TrilinosWrappers::PreconditionBase> preconditioner_C;

        std::shared_ptr<TrilinosWrappers::PreconditionBase> preconditioner_S;

        mutable TrilinosWrappers::MPI::BlockVector tmp;
        
        mutable TrilinosWrappers::MPI::Vector tmp_2;

        unsigned int maxit;

        double tol;
    };

    /*****************************************************************/
    /*                 END PRECONDITIONERS SECTION                   */
    /*****************************************************************/

    // --------------------------------PUBLIC FUNCTIONS--------------------------------

    MonolithicNavierStokes(
        const std::string &mesh_file_name_,
        const unsigned int &degree_velocity_,
        const unsigned int &degree_pressure_,
        const double &T_,
        const double &deltat_,
        const double &re_)
        : mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD))
        , mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
        , pcout(std::cout, mpi_rank == 0)
        , mesh_file_name(mesh_file_name_)
        , degree_velocity(degree_velocity_)
        , degree_pressure(degree_pressure_)
        , T(T_)
        , deltat(deltat_)
        , inlet_velocity(H)
        , velocity(0)
        , pressure(dim)
        , mesh(MPI_COMM_WORLD)
        , reynolds_number(re_)
    {
        this->nu = (2. / 3.) * inlet_velocity.get_u_max() * cylinder_radius / reynolds_number;
    }

    // Function to run the full problem pipeline.
    void
    run();

    // ------------------------------PRIVATE FUNCTIONS-------------------------------

private:
    // Setup the problem.
    void
    setup();

    // Assemble the time independent part of the system matrix.
    void 
    assemble_base_matrix();

    // Assemble the convective part of the system matrix.
    void
    add_convective_term();

    // Assemble the right-hand side of the problem.
    void
    assemble_rhs();

    // Group all the instructions that need to be executed at each time step.
    void
    solve_time_step();

    // Solve the entire problem by looping over time steps.
    void
    solve();

    // Save the output of the computation in a pvtk format.
    void
    output(const unsigned int &time_step);

    // Defines the path of the directory where the outputs will be stored
    std::string
    get_output_directory();

    // --------------------------------PRIVATE VARIABLES----------------------------------

    // Number of MPI processes.
    const unsigned int mpi_size;

    // This MPI process.
    const unsigned int mpi_rank;

    // Parallel output stream.
    ConditionalOStream pcout;

    // Height of the channel.
    static constexpr double H = 0.41;

    // Cylinder radius.
    static constexpr double cylinder_radius = 0.1;

    // Mesh file name.
    const std::string mesh_file_name;

    // Polynomial degree Velocity.
    const unsigned int degree_velocity;

    // Polynomial degree Pressure.
    const unsigned int degree_pressure;

    // Final time.
    const double T;

    // Time step.
    const double deltat;

    // Inlet velocity.
    InletVelocity inlet_velocity;

    // extractors for velocity.
    FEValuesExtractors::Vector velocity;

    // extractors for pressure.
    FEValuesExtractors::Scalar pressure;

    // Mesh.
    parallel::fullydistributed::Triangulation<dim> mesh;

    // Reynolds number.
    const double reynolds_number;

    // Affine constraints.
    AffineConstraints<double> constraints;

    // diffusion coefficient
    double nu;

    // Forcing term.
    Functions::ZeroFunction<dim> forcing_term;

    // Function for neuman condition
    Functions::ZeroFunction<dim> neumann_function;

    // Function for the initial condition
    Functions::ZeroFunction<dim> initial_condition;

    // Current time.
    double time;

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

    // Block-wise partitioning of DoFs owned by the current process.
    std::vector<IndexSet> block_owned_dofs;

    // DoFs relevant to the current process (including ghost DoFs).
    IndexSet locally_relevant_dofs;

    // Block-wise partitioning of DoFs relevant to the current process.
    std::vector<IndexSet> block_relevant_dofs;

    // Time independent matrix
    TrilinosWrappers::BlockSparseMatrix system_matrix;

    // velocity mass.
    TrilinosWrappers::BlockSparseMatrix velocity_mass;

    // Complete matrix.
    TrilinosWrappers::BlockSparseMatrix lhs_matrix;

    TrilinosWrappers::BlockSparseMatrix pressure_mass;

    // Right-hand side vector in the linear system.
    TrilinosWrappers::MPI::BlockVector system_rhs;

    // System solution (without ghost elements).
    TrilinosWrappers::MPI::BlockVector solution_owned;

    // System solution (including ghost elements).
    TrilinosWrappers::MPI::BlockVector solution;

};

#endif