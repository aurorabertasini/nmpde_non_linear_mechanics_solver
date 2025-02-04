#ifndef NAVIERSTOKES_HPP
#define NAVIERSTOKES_HPP

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/distributed/fully_distributed_tria.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/trilinos_block_sparse_matrix.h>
#include <deal.II/lac/trilinos_parallel_block_vector.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values_extractors.h>
#include <deal.II/fe/mapping_fe.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

// ---------------------------------------------------------------------

#include <filesystem>
#include <fstream>
#include <iostream>

using namespace dealii;

template <unsigned int dim>
class MonolithicNavierStokes
{
public:
    class ForcingTerm2D : public Function<dim>
    {
    public:
        virtual void
        vector_value(const Point<dim> &p,
                     Vector<double> &values) const override
        {
            double x = p[0];
            double y = p[1];
            double t = this->get_time();
            values[0] = M_PI * (4.0 * M_PI * M_PI * pow(sin(t), 2) * pow(sin(M_PI * x), 3) * sin(M_PI * y) * cos(M_PI * x) + 16.0 * M_PI * M_PI * sin(t) * pow(sin(M_PI * x), 2) * cos(M_PI * y) - sin(t) * sin(M_PI * x) - 4.0 * M_PI * M_PI * sin(t) * cos(M_PI * y) + 2.0 * pow(sin(M_PI * x), 2) * cos(t) * cos(M_PI * y)) * sin(M_PI * y);

            values[1] = M_PI * (4.0 * M_PI * M_PI * pow(sin(t), 2) * pow(sin(M_PI * x), 2) * pow(sin(M_PI * y), 3) * cos(M_PI * y) - 16.0 * M_PI * M_PI * sin(t) * sin(M_PI * x) * pow(sin(M_PI * y), 2) * cos(M_PI * x) + 4.0 * M_PI * M_PI * sin(t) * sin(M_PI * x) * cos(M_PI * x) + sin(t) * cos(M_PI * x) * cos(M_PI * y) - 2.0 * sin(M_PI * x) * pow(sin(M_PI * y), 2) * cos(t) * cos(M_PI * x));
        }

        virtual double
        value(const Point<dim> &p,
              const unsigned int component = 0) const override
        {
            double x = p[0];
            double y = p[1];
            double t = this->get_time();
            if (component == 0)
                return M_PI * (4.0 * M_PI * M_PI * pow(sin(t), 2) * pow(sin(M_PI * x), 3) * sin(M_PI * y) * cos(M_PI * x) + 16.0 * M_PI * M_PI * sin(t) * pow(sin(M_PI * x), 2) * cos(M_PI * y) - sin(t) * sin(M_PI * x) - 4.0 * M_PI * M_PI * sin(t) * cos(M_PI * y) + 2.0 * pow(sin(M_PI * x), 2) * cos(t) * cos(M_PI * y)) * sin(M_PI * y);
            else if (component == 1)
                return M_PI * (4.0 * M_PI * M_PI * pow(sin(t), 2) * pow(sin(M_PI * x), 2) * pow(sin(M_PI * y), 3) * cos(M_PI * y) - 16.0 * M_PI * M_PI * sin(t) * sin(M_PI * x) * pow(sin(M_PI * y), 2) * cos(M_PI * x) + 4.0 * M_PI * M_PI * sin(t) * sin(M_PI * x) * cos(M_PI * x) + sin(t) * cos(M_PI * x) * cos(M_PI * y) - 2.0 * sin(M_PI * x) * pow(sin(M_PI * y), 2) * cos(t) * cos(M_PI * x));
            else
                return 0.0;
        }

    protected:
    };

    // Class for the exact solution, containing both velocity and pressure.
    class ExactSolution2D : public Function<dim>
    {
    public:
        // Constructor.
        ExactSolution2D()
            : Function<dim>(dim + 1),
              exact_velocity(),
              exact_pressure() {}

        // When defining vector-valued functions, we need to define the value
        // function, which returns the value of the function at a given point and
        // component...
        virtual double value(const Point<dim> &p,
                             const unsigned int component) const override
        {
            if (component < dim)
            {
                exact_velocity.set_time(this->get_time());
                return exact_velocity.value(p, component);
            }
            else if (component == dim)
            {
                exact_pressure.set_time(this->get_time());
                return exact_pressure.value(p);
            }
            else
            {
                return 0.0;
            }
        }

        // ... and the vector_value function, which returns the value of the
        // function at a given point for all components.
        virtual void vector_value(const Point<dim> &p,
                                  Vector<double> &values) const override
        {
            for (unsigned int i = 0; i < dim + 1; i++)
            {
                values[i] = value(p, i);
            }
        }

        // This is a function object, which defines the exact velocity. Since the
        // problem's exact solution is known, we can define it as a function object
        // and use it to compute the error of our numerical solution. To be able to
        // compute the H1 norm of the error, the exact gradient is computed as well.
        // This function returns 4 values, despite the fact that the last one is
        // empty, for compatibility reasons.
        class ExactVelocity2D : public Function<dim>
        {
        public:
            virtual void
            vector_value(const Point<dim> &p, Vector<double> &values) const override
            {
                double t = this->get_time();
                // u_x = sp.pi * sp.sin(t) * sp.sin(2 * sp.pi * y) * sp.sin(sp.pi * x)**2
                values[0] = M_PI * sin(t) * sin(2.0 * M_PI * p[1]) * pow(sin(M_PI * p[0]), 2);

                // u_y = -sp.pi * sp.sin(t) * sp.sin(2 * sp.pi * x) * sp.sin(sp.pi * y)**2
                values[1] = -M_PI * sin(t) * sin(2.0 * M_PI * p[0]) * pow(sin(M_PI * p[1]), 2);
            }

            virtual double
            value(const Point<dim> &p, const unsigned int component = 0) const override
            {
                double t = this->get_time();
                if (component == 0)
                    return M_PI * sin(t) * sin(2.0 * M_PI * p[1]) * pow(sin(M_PI * p[0]), 2);

                else if (component == 1)
                    return -M_PI * sin(t) * sin(2.0 * M_PI * p[0]) * pow(sin(M_PI * p[1]), 2);
                else
                    return 0.0;
            }

            virtual Tensor<1, dim>
            gradient(
                const Point<dim> &p, const unsigned int component) const
            {
                Tensor<1, dim> result;
                double t = this->get_time();
                double x = p[0];
                double y = p[1];

                if (component == 0)
                {
                    result[0] = 2.0 * M_PI * M_PI * sin(t) * sin(2.0 * M_PI * y) * sin(M_PI * x) * cos(M_PI * x);
                    result[1] = 2.0 * M_PI * M_PI * sin(t) * cos(2.0 * M_PI * y) * pow(sin(M_PI * x), 2);
                }
                else if (component == 1)
                {
                    result[0] = -2.0 * M_PI * M_PI * sin(t) * cos(2.0 * M_PI * x) * pow(sin(M_PI * y), 2);
                    result[1] = -2.0 * M_PI * M_PI * sin(t) * sin(2.0 * M_PI * x) * sin(M_PI * y) * cos(M_PI * y);
                }
                return result;
            }
            virtual void vector_gradient(
                const Point<dim> &p, std::vector<Tensor<1, dim>> &values) const
            {
                for (unsigned int i = 0; i < dim; i++)
                {
                    values[i] = gradient(p, i);
                }
            }
        };

        class ExactPressure2D : public Function<dim>
        {
        public:
            virtual void
            vector_value(const Point<dim> &p, Vector<double> &values) const override
            {
                // p=sp.sin(t) * sp.cos(sp.pi * x) * sp.sin(sp.pi * y)

                values[0] = sin(this->get_time()) * cos(M_PI * p[0]) * sin(M_PI * p[1]);

                for (unsigned int i = 1; i < dim + 1; ++i)
                    values[i] = 0.0;
            }

            virtual double
            value(const Point<dim> &p, const unsigned int component = 0) const override
            {
                if (component == 0)
                    return sin(this->get_time()) * cos(M_PI * p[0]) * sin(M_PI * p[1]);
                else
                    return 0.0;
            }
        };

    private:
    public:
        mutable ExactVelocity2D exact_velocity;

        mutable ExactPressure2D exact_pressure;
    };

    class EthierSteinman : public Function<dim>
    {
    public:
        EthierSteinman(double nu_) : Function<dim>(dim + 1), exact_velocity(nu_), exact_pressure(nu_), neumann_condition(nu_)
        {
        }

        virtual void 
        set_time(const double time) override
        {
            Function<dim>::set_time(time);
            exact_velocity.set_time(time);
            exact_pressure.set_time(time);
            neumann_condition.set_time(time);
        }

        virtual double
        value(const Point<dim> &p, const unsigned int component) const
        {
            if (component < dim)
            {
                exact_velocity.set_time(this->get_time());
                return exact_velocity.value(p, component);
            }
            else if (component == dim)
            {
                exact_pressure.set_time(this->get_time());
                return exact_pressure.value(p);
            }
            else
            {
                return 0.0;
            }
        }

        virtual void
        vector_value(const Point<dim> &p, Vector<double> &values) const override
        {
            for (unsigned int i = 0; i < dim + 1; i++)
            {
                values[i] = value(p, i);
            }
        }
        class EthierSteinmanVelocity : public Function<dim>
        {
        public:
            EthierSteinmanVelocity(double nu_)
                : Function<dim>(dim), nu(nu_)
            {
            }

            virtual void
            vector_value(const Point<dim> &p, Vector<double> &values) const override
            {
                double t = this->get_time();
                double x1 = p[0];
                double x2 = p[1];
                double x3 = p[2];

                double factor = -a * std::exp(-nu * b * b * t);

                values[0] = factor * (std::exp(a * x1) * std::sin(a * x2 + b * x3) + std::exp(a * x3) * std::cos(a * x1 + b * x2));
                values[1] = factor * (std::exp(a * x2) * std::sin(a * x3 + b * x1) + std::exp(a * x1) * std::cos(a * x2 + b * x3));
                values[2] = factor * (std::exp(a * x3) * std::sin(a * x1 + b * x2) + std::exp(a * x2) * std::cos(a * x3 + b * x1));
                values[3] = 0.0;
            }

            virtual double
            value(const Point<dim> &p, const unsigned int component = 0) const override
            {
                double t = this->get_time();
                double x1 = p[0];
                double x2 = p[1];
                double x3 = p[2];

                double factor = -a * std::exp(-nu * b * b * t);

                if (component == 0)
                    return factor * (std::exp(a * x1) * std::sin(a * x2 + b * x3) + std::exp(a * x3) * std::cos(a * x1 + b * x2));
                else if (component == 1)
                    return factor * (std::exp(a * x2) * std::sin(a * x3 + b * x1) + std::exp(a * x1) * std::cos(a * x2 + b * x3));
                else if (component == 2)
                    return factor * (std::exp(a * x3) * std::sin(a * x1 + b * x2) + std::exp(a * x2) * std::cos(a * x3 + b * x1));
                else 
                return 0.0;

            }

            virtual Tensor<1, dim>
            gradient(
                const Point<dim> &p, const unsigned int component) const
            {
                Tensor<1, dim> result;

                for (unsigned int i = 0; i < dim; i++)
                {
                    result[i] = -a * std::exp(-nu * b * b * this->get_time());
                }
                if (component == 0)
                {
                    result[0] *= (a * std::exp(a * p[0]) * std::sin(a * p[1] + b * p[2]) -
                                  a * std::exp(a * p[2]) * std::sin(a * p[0] + b * p[1]));
                    result[1] *= (a * std::exp(a * p[0]) * std::cos(a * p[1] + b * p[2]) -
                                  b * std::exp(a * p[2]) * std::sin(a * p[0] + b * p[1]));
                    result[2] *= (b * std::exp(a * p[0]) * std::cos(a * p[1] + b * p[2]) +
                                  a * std::exp(a * p[2]) * std::cos(a * p[0] + b * p[1]));
                }
                else if (component == 1)
                {
                    result[0] *= (b * std::exp(a * p[1]) * std::cos(a * p[2] + b * p[0]) +
                                  a * std::exp(a * p[0]) * std::cos(a * p[1] + b * p[2]));
                    result[1] *= (a * std::exp(a * p[1]) * std::sin(a * p[2] + b * p[0]) -
                                  a * std::exp(a * p[0]) * std::sin(a * p[1] + b * p[2]));
                    result[2] *= (a * std::exp(a * p[1]) * std::cos(a * p[2] + b * p[0]) -
                                  b * std::exp(a * p[0]) * std::sin(a * p[1] + b * p[2]));
                }
                else if (component == 2)
                {
                    result[0] *= (a * std::exp(a * p[2]) * std::cos(a * p[0] + b * p[1]) -
                                  b * std::exp(a * p[1]) * std::sin(a * p[2] + b * p[0]));
                    result[1] *= (b * std::exp(a * p[2]) * std::cos(a * p[0] + b * p[1]) +
                                  a * std::exp(a * p[1]) * std::cos(a * p[2] + b * p[0]));
                    result[2] *= (a * std::exp(a * p[2]) * std::sin(a * p[0] + b * p[1]) -
                                  a * std::exp(a * p[1]) * std::sin(a * p[2] + b * p[0]));
                }
                else
                {
                    for (unsigned int i = 0; i < dim; i++)
                    {
                        result[i] = 0.0;
                    }
                }

                return result;
            }

            virtual void vector_gradient(
                const Point<dim> &p, std::vector<Tensor<1, dim>> &values) const
            {
                for (unsigned int i = 0; i < dim + 1; i++)
                {
                    values[i] = gradient(p, i);
                }
            }

        private:
            static constexpr double a = M_PI / 4.;
            static constexpr double b = M_PI / 2.;
            const double nu;
        };

        class EthierSteinmanPressure : public Function<dim>
        {
        public:
            EthierSteinmanPressure(double nu_)
                : Function<dim>(1), nu(nu_)
            {
            }

            virtual double
            value(const Point<dim> &p, const unsigned int /*component*/ = 0) const override
            {
                double t = this->get_time();
                return -a * a / 2.0 * std::exp(-2 * nu * b * b * t) *
                       (2.0 * std::sin(a * p[0] + b * p[1]) * std::cos(a * p[2] + b * p[0]) *
                            std::exp(a * (p[1] + p[2])) +
                        2.0 * std::sin(a * p[1] + b * p[2]) * std::cos(a * p[0] + b * p[1]) *
                            std::exp(a * (p[0] + p[2])) +
                        2.0 * std::sin(a * p[2] + b * p[0]) * std::cos(a * p[1] + b * p[2]) *
                            std::exp(a * (p[0] + p[1])) +
                        std::exp(2.0 * a * p[0]) + std::exp(2.0 * a * p[1]) +
                        std::exp(2.0 * a * p[2]));
            }

            virtual void
            vector_value(const Point<dim> &p, Vector<double> &values) const override
            {
                double t = this->get_time();
                values[0] = -a * a / 2.0 * std::exp(-2 * nu * b * b * t) *
                            (2.0 * std::sin(a * p[0] + b * p[1]) * std::cos(a * p[2] + b * p[0]) *
                                 std::exp(a * (p[1] + p[2])) +
                             2.0 * std::sin(a * p[1] + b * p[2]) * std::cos(a * p[0] + b * p[1]) *
                                 std::exp(a * (p[0] + p[2])) +
                             2.0 * std::sin(a * p[2] + b * p[0]) * std::cos(a * p[1] + b * p[2]) *
                                 std::exp(a * (p[0] + p[1])) +
                             std::exp(2.0 * a * p[0]) + std::exp(2.0 * a * p[1]) +
                             std::exp(2.0 * a * p[2]));
            }

        private:
            static constexpr double a = M_PI / 4.;
            static constexpr double b = M_PI / 2.;
            const double nu;
        };

        class EthierSteinmanNeumann : public Function<dim>
        {
        public:
            EthierSteinmanNeumann(double nu_)
                : Function<dim>(dim + 1), nu(nu_), exact_velocity_n(nu_), exact_pressure_n(nu_)
            {
            }

            virtual double
            value(const Point<dim> &p, const unsigned int component) const
            {
                exact_pressure_n.set_time(this->get_time());
                exact_velocity_n.set_time(this->get_time());

                if (component == 0 || component == 2)
                {
                    Tensor<1, dim> velocity_gradient =
                        exact_velocity_n.gradient(p, component);

                    return -nu * velocity_gradient[1];
                }
                else if (component == 1)
                {
                    Tensor<1, dim> velocity_gradient =
                        exact_velocity_n.gradient(p, component);
                    return -nu * velocity_gradient[1] + exact_pressure_n.value(p);
                }
                else
                {
                    return 0.0;
                }
            }

            virtual void
            vector_value(const Point<dim> &p, Vector<double> &values) const override
            {
                for (unsigned int i = 0; i < dim; i++)
                {
                    values[i] = value(p, i);
                }
                values[dim] = 0.0;
            }

        private:
            mutable EthierSteinmanVelocity exact_velocity_n;
            mutable EthierSteinmanPressure exact_pressure_n;

            const double nu;
        };

        mutable EthierSteinmanVelocity exact_velocity;
        mutable EthierSteinmanPressure exact_pressure;
        mutable EthierSteinmanNeumann neumann_condition;
    };

    /*****************************************************************/
    /*****************************************************************/
    /*         PRECONDITIONERS OPTIONS                               */
    /*****************************************************************/
    /*****************************************************************/

    // Abstract class for block preconditioners

    class BlockPrecondition
    {
    public:
        virtual ~BlockPrecondition() = default;
        virtual void vmult(TrilinosWrappers::MPI::BlockVector &dst,
                           const TrilinosWrappers::MPI::BlockVector &src) const = 0;
    };

    /**
     * Block-diagonal preconditioner for Navier–Stokes or Stokes:
     * Each diagonal block (velocity, pressure) uses an inner preconditioner
     * that can be ILU or AMG. We provide two “initialize” methods.
     */

    class PreconditionBlockDiagonal : public BlockPrecondition
    {
    public:
        /**
         * ILU-based initialization for both velocity and pressure blocks.
         */
        //   // Block-diagonal preconditioner.
        // Adapted from the one proposed for the Stokes problem in laboratory 9.
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

        // Initialize the preconditioner.
        void initialize(const TrilinosWrappers::SparseMatrix &velocity_stiffness_,
                        const TrilinosWrappers::SparseMatrix &pressure_mass_,
                        const unsigned int &maxit_, const double &tol_,
                        const bool &use_ilu)
        {
            maxit = maxit_;
            tol = tol_;
            // Save a reference to the input matrices.
            velocity_stiffness = &velocity_stiffness_;
            pressure_mass = &pressure_mass_;

            // Initialize the preconditioners.
            initialize_inner_preconditioner(preconditioner_velocity, *velocity_stiffness,
                                            use_ilu);
            initialize_inner_preconditioner(preconditioner_pressure, *pressure_mass,
                                            use_ilu);
        }

        // Application of the preconditioner.
        void vmult(TrilinosWrappers::MPI::BlockVector &dst,
                   const TrilinosWrappers::MPI::BlockVector &src) const override

        {
            // Solve the top-left block.
            SolverControl solver_control_velocity(maxit, tol * src.block(0).l2_norm());
            SolverGMRES<TrilinosWrappers::MPI::Vector> solver_gmres_velocity(
                solver_control_velocity);
            solver_gmres_velocity.solve(*velocity_stiffness, dst.block(0), src.block(0),
                                        *preconditioner_velocity);

            // Solve the bottom-right block.
            SolverControl solver_control_pressure(maxit, tol * src.block(1).l2_norm());
            SolverCG<TrilinosWrappers::MPI::Vector> solver_cg_pressure(
                solver_control_pressure);
            solver_cg_pressure.solve(*pressure_mass, dst.block(1), src.block(1),
                                     *preconditioner_pressure);
        }

    private:
        // Velocity stiffness matrix.
        const TrilinosWrappers::SparseMatrix *velocity_stiffness;

        // Preconditioner used for the velocity block.
        std::shared_ptr<TrilinosWrappers::PreconditionBase> preconditioner_velocity;

        // Pressure mass matrix.
        const TrilinosWrappers::SparseMatrix *pressure_mass;

        // Preconditioner used for the pressure block.
        std::shared_ptr<TrilinosWrappers::PreconditionBase> preconditioner_pressure;

        // Maximum number of iterations for the inner solvers.
        unsigned int maxit;

        // Tolerance for the inner solvers.
        double tol;
    };

    // SIMPLE preconditioner.
    class PreconditionSIMPLE : public BlockPrecondition
    {
    public:
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

        // Initialize the preconditioner.
        void initialize(const TrilinosWrappers::SparseMatrix &F_matrix_,
                        const TrilinosWrappers::SparseMatrix &negB_matrix_,
                        const TrilinosWrappers::SparseMatrix &Bt_matrix_,
                        const TrilinosWrappers::MPI::BlockVector &vec,
                        const double &alpha_, const unsigned int &maxit_,
                        const double &tol_, const bool &use_ilu)
        {
            alpha = alpha_;
            maxit = maxit_;
            tol = tol_;
            // Save a reference to the input matrices.
            F_matrix = &F_matrix_;
            negB_matrix = &negB_matrix_;
            Bt_matrix = &Bt_matrix_;

            // Save the negated inverse diagonal of F.
            negDinv_vector.reinit(vec.block(0));
            for (unsigned int index : negDinv_vector.locally_owned_elements())
            {
                negDinv_vector[index] = -1.0 / F_matrix->diag_element(index);
            }

            // Create the matrix S.
            negB_matrix->mmult(S_matrix, *Bt_matrix, negDinv_vector);

            // Initialize the preconditioners.
            initialize_inner_preconditioner(preconditioner_F, *F_matrix, use_ilu);
            initialize_inner_preconditioner(preconditioner_S, S_matrix, use_ilu);
        }

        // Application of the preconditioner.
        void vmult(
            TrilinosWrappers::MPI::BlockVector &dst,
            const TrilinosWrappers::MPI::BlockVector &src) const
        {
            tmp.reinit(src);
            // Step 1: solve [F 0; B -S]sol1 = src.
            // Step 1.1: solve F*sol1_u = src_u.
            SolverControl solver_control_F(maxit, tol);
            SolverGMRES<TrilinosWrappers::MPI::Vector> solver_F(solver_control_F);
            solver_F.solve(*F_matrix, tmp.block(0), src.block(0), *preconditioner_F);
            // Step 1.2: solve S*sol1_p = B*sol1_u - src_p.
            Bt_matrix->Tvmult(tmp.block(1), tmp.block(0));
            tmp.block(1) -= src.block(1);
            SolverControl solver_control_S(maxit, tol);
            SolverGMRES<TrilinosWrappers::MPI::Vector> solver_S(solver_control_S);
            solver_S.solve(S_matrix, dst.block(1), tmp.block(1), *preconditioner_S);

            // Step 2: solve [I D^-1*B^T; 0 alpha*I]dst = sol1.
            // Step 2.1: solve alpha*I*dst_p = sol1_p.
            dst.block(1) /= alpha;
            // Step 2.2: solve dst_u = sol1_u - D^-1*B^T*dst_p.
            dst.block(0) = tmp.block(0);
            Bt_matrix->vmult(tmp.block(0), dst.block(1));
            tmp.block(0).scale(negDinv_vector);
            dst.block(0) += tmp.block(0);
        }

    private:
        // Damping parameter (must be in (0,1]).
        double alpha;

        // Matrix F (top left block of the system matrix).
        const TrilinosWrappers::SparseMatrix *F_matrix;

        // Matrix -B (bottom left block of the system matrix).
        const TrilinosWrappers::SparseMatrix *negB_matrix;

        // Matrix B^T (top right block of the system matrix).
        const TrilinosWrappers::SparseMatrix *Bt_matrix;

        // Matrix -D^-1, negative inverse diagonal of F.
        TrilinosWrappers::MPI::Vector negDinv_vector;

        // Matrix S := B*D^-1*B^T.
        TrilinosWrappers::SparseMatrix S_matrix;

        // Preconditioner used for the block multiplied by F.
        std::shared_ptr<TrilinosWrappers::PreconditionBase> preconditioner_F;

        // Preconditioner used for the block multiplied by S.
        std::shared_ptr<TrilinosWrappers::PreconditionBase> preconditioner_S;

        // Temporary vector.
        mutable TrilinosWrappers::MPI::BlockVector tmp;

        // Maximum number of iterations for the inner solvers.
        unsigned int maxit;

        // Tolerance for the inner solvers.
        double tol;
    };

    // aSIMPLE preconditioner.
    class PreconditionaSIMPLE : public BlockPrecondition
    {
    public:
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

        void initialize(
            const TrilinosWrappers::SparseMatrix &F_matrix_,
            const TrilinosWrappers::SparseMatrix &negB_matrix_,
            const TrilinosWrappers::SparseMatrix &Bt_matrix_,
            const TrilinosWrappers::MPI::BlockVector &vec, const double &alpha_,
            const bool &use_inner_solver_, const unsigned int &maxit_,
            const double &tol_, const bool &use_ilu)
        {
            alpha = alpha_;
            use_inner_solver = use_inner_solver_;
            maxit = maxit_;
            tol = tol_;
            // Save a reference to the input matrices.
            F_matrix = &F_matrix_;
            negB_matrix = &negB_matrix_;
            Bt_matrix = &Bt_matrix_;

            // Save the diagonal and inverse diagonal of F.
            D_vector.reinit(vec.block(0));
            Dinv_vector.reinit(vec.block(0));
            for (unsigned int index : D_vector.locally_owned_elements())
            {
                const double value = F_matrix->diag_element(index);
                D_vector[index] = value;
                Dinv_vector[index] = 1.0 / value;
            }

            // Create the matrix -S.
            negB_matrix->mmult(negS_matrix, *Bt_matrix, Dinv_vector);

            // Initialize the preconditioners.
            initialize_inner_preconditioner(preconditioner_F, *F_matrix, use_ilu);
            initialize_inner_preconditioner(preconditioner_S, negS_matrix, use_ilu);
        }

        // Application of the preconditioner.
        void vmult(TrilinosWrappers::MPI::BlockVector &dst,
                   const TrilinosWrappers::MPI::BlockVector &src) const
        {
            tmp.reinit(src);
            // Step 1: multiply src by [F^-1 0; 0 I].
            if (use_inner_solver)
            {
                SolverControl solver_control_F(maxit, tol * src.block(0).l2_norm());
                SolverGMRES<TrilinosWrappers::MPI::Vector> solver_F(solver_control_F);
                solver_F.solve(*F_matrix, dst.block(0), src.block(0), *preconditioner_F);
            }
            else
            {
                preconditioner_F->vmult(dst.block(0), src.block(0));
            }
            tmp.block(1) = src.block(1);
            // Step 2: multiply the result by [I 0; -B I].
            negB_matrix->vmult_add(tmp.block(1), dst.block(0));
            // Step 3: multiply the result by [I 0; 0 -S^-1].
            if (use_inner_solver)
            {
                SolverControl solver_control_S(maxit, tol * tmp.block(1).l2_norm());
                SolverGMRES<TrilinosWrappers::MPI::Vector> solver_S(solver_control_S);
                solver_S.solve(negS_matrix, dst.block(1), tmp.block(1), *preconditioner_S);
            }
            else
            {
                preconditioner_S->vmult(dst.block(1), tmp.block(1));
            }
            // Step 4: multiply the result by [D 0; 0 I/alpha].
            dst.block(0).scale(D_vector);
            dst.block(1) /= alpha;
            // Step 5: multiply the result by [I -B^T; 0 I].
            Bt_matrix->vmult(tmp.block(0), dst.block(1));
            dst.block(0) -= tmp.block(0);
            // Step 6: multiply the result by [D^-1 0; 0 I].
            dst.block(0).scale(Dinv_vector);
        }

    private:
        // Damping parameter (must be in (0,1]).
        double alpha;

        // Matrix F (top left block of the system matrix).
        const TrilinosWrappers::SparseMatrix *F_matrix;

        // Matrix -B (bottom left block of the system matrix).
        const TrilinosWrappers::SparseMatrix *negB_matrix;

        // Matrix B^T (top right block of the system matrix).
        const TrilinosWrappers::SparseMatrix *Bt_matrix;

        // Matrix D, diagonal of F.
        TrilinosWrappers::MPI::Vector D_vector;

        // Matrix D^-1.
        TrilinosWrappers::MPI::Vector Dinv_vector;

        // Matrix -S := -B*D^-1*B^T.
        TrilinosWrappers::SparseMatrix negS_matrix;

        // Preconditioner used for the block multiplied by F.
        std::shared_ptr<TrilinosWrappers::PreconditionBase> preconditioner_F;

        // Preconditioner used for the block multiplied by S.
        std::shared_ptr<TrilinosWrappers::PreconditionBase> preconditioner_S;

        // Temporary vector.
        mutable TrilinosWrappers::MPI::BlockVector tmp;

        // Whether to use inner solvers.
        bool use_inner_solver;

        // Maximum number of iterations for the inner solvers.
        unsigned int maxit;

        // Tolerance for the inner solvers.
        double tol;
    };

    // Yosida preconditioner.
    class PreconditionYosida : public BlockPrecondition
    {
    public:
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

        // Initialize the preconditioner.
        void initialize(const TrilinosWrappers::SparseMatrix &F_matrix_,
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
            F_matrix = &F_matrix_;
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
            initialize_inner_preconditioner(preconditioner_F, *F_matrix, use_ilu);
            initialize_inner_preconditioner(preconditioner_S, negS_matrix, use_ilu);
        }

        // Application of the preconditioner.
        void vmult(TrilinosWrappers::MPI::BlockVector &dst,
                   const TrilinosWrappers::MPI::BlockVector &src) const
        {
            tmp.reinit(src);
            // Step 1: solve [F 0; B -S]sol1 = src.
            // Step 1.1: solve F*sol1_u = src_u.
            tmp.block(0) = dst.block(0);
            SolverControl solver_control_F(maxit, tol * src.block(0).l2_norm());
            SolverGMRES<TrilinosWrappers::MPI::Vector> solver_F(solver_control_F);
            solver_F.solve(*F_matrix, tmp.block(0), src.block(0), *preconditioner_F);
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
            SolverControl solver_control_F2(maxit, tol * tmp.block(0).l2_norm());
            SolverGMRES<TrilinosWrappers::MPI::Vector> solver_gmres_F2(solver_control_F);
            solver_gmres_F2.solve(*F_matrix, tmp_2, tmp.block(0), *preconditioner_F);
            dst.block(0) -= tmp_2;
        }

    private:
        // Matrix F (top left block of the system matrix).
        const TrilinosWrappers::SparseMatrix *F_matrix;

        // Matrix -B (bottom left block of the system matrix).
        const TrilinosWrappers::SparseMatrix *negB_matrix;

        // Matrix B^T (top right block of the system matrix).
        const TrilinosWrappers::SparseMatrix *Bt_matrix;

        // Matrix D^-1, inverse diagonal of M/deltat.
        TrilinosWrappers::MPI::Vector Dinv_vector;

        // Matrix -S := -B*D^-1*B^T.
        TrilinosWrappers::SparseMatrix negS_matrix;

        // Preconditioner used for the block multiplied by F.
        std::shared_ptr<TrilinosWrappers::PreconditionBase> preconditioner_F;

        // Preconditioner used for the block multiplied by S.
        std::shared_ptr<TrilinosWrappers::PreconditionBase> preconditioner_S;

        // Temporary vectors.
        mutable TrilinosWrappers::MPI::BlockVector tmp;
        mutable TrilinosWrappers::MPI::Vector tmp_2;

        // Maximum number of iterations for the inner solvers.
        unsigned int maxit;

        // Tolerance for the inner solvers.
        double tol;
    };

    /*****************************************************************/
    /*****************************************************************/
    /*         END PRECONDITIONERS SECTION                           */
    /*****************************************************************/
    /*****************************************************************/

    // parameter as constructor arguments.
    MonolithicNavierStokes(
        const std::string &mesh_file_name_,
        const unsigned int &degree_velocity_,
        const unsigned int &degree_pressure_,
        const double &T_,
        const double &deltat_);
    // Initialization.
    void
    setup();

    // Solve the problem.
    void
    solve();

    // Compute the error.
    double
    compute_error(const VectorTools::NormType &norm_type, bool velocity);

    std::string
    get_output_directory();

    double get_l2_H1_error();

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
    compute_lift_drag();

    void
    update_buondary_conditions();

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
    const double nu = 1.0;

    // Inlet velocity.

    // // Exact solution.
    // ExactSolution exact_solution;

    // Current time.
    double time;

    // Final time.
    const double T;

    double lift;

    double drag;

    double l2_H1_error = 0.0;
    // Discretization. ///////////////////////////////////////////////////////////

    // Mesh file name.
    const std::string mesh_file_name;

    // Polynomial degree Velocity.
    const unsigned int degree_velocity;

    // Polynomial degree Pressure.
    const unsigned int degree_pressure;

    // Time step.
    const double deltat;

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

    // velocity mass
    TrilinosWrappers::BlockSparseMatrix velocity_mass;

    // Non_linear term matrix.
    TrilinosWrappers::BlockSparseMatrix convective_matrix;

    // boolean value to choose to reconstruct all the system matrix ad each time step
    bool RECONSTRUCT_SYSTEM_MATRIX = true;

    // ExactSolution2D exact_solution;

    EthierSteinman exact_solution3D;

    // forcing term 2D
    ForcingTerm2D forcing_term;

    // Stiffness matrix K.
    TrilinosWrappers::BlockSparseMatrix stiffness_matrix;

    TrilinosWrappers::BlockSparseMatrix pressure_mass; //!

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

template <unsigned int dim>
MonolithicNavierStokes<dim>::MonolithicNavierStokes(
    const std::string &mesh_file_name_,
    const unsigned int &degree_velocity_,
    const unsigned int &degree_pressure_,
    const double &T_,
    const double &deltat_)
    : mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD))
    , mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
    , pcout(std::cout, mpi_rank == 0)
    , mesh_file_name(mesh_file_name_)
    , degree_velocity(degree_velocity_)
    , degree_pressure(degree_pressure_)
    , T(T_), deltat(deltat_)
    , velocity(0)
    , pressure(dim)
    , mesh(MPI_COMM_WORLD)
    , exact_solution3D(nu)
{
}

#endif