#ifndef MONOLITHICNAVIERSTOKES_HPP
#define MONOLITHICNAVIERSTOKES_HPP

#include "includes_file.hpp"
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
            values[0] = M_PI * (4.0 * M_PI * M_PI * sin(t) * sin(t) * sin(M_PI * x) * sin(M_PI * x) 
                                    * sin(M_PI * x) * sin(M_PI * y) * cos(M_PI * x) + 16.0 * M_PI 
                                    * M_PI * sin(t) * sin(M_PI * x) * sin(M_PI * x) * cos(M_PI * y) 
                                    - sin(t) * sin(M_PI * x) - 4.0 * M_PI * M_PI * sin(t) * cos(M_PI * y) 
                                    + 2.0 * sin(M_PI * x) * sin(M_PI * x) * cos(t) * cos(M_PI * y)) * sin(M_PI * y);

            values[1] = M_PI * (4.0 * M_PI * M_PI * sin(t) * sin(t) * sin(M_PI * x) * sin(M_PI * x) 
                                    * sin(M_PI * y) * sin(M_PI * y) * sin(M_PI * y) * cos(M_PI * y) 
                                    - 16.0 * M_PI * M_PI * sin(t) * sin(M_PI * x) * sin(M_PI * y) 
                                    * sin(M_PI * y) * cos(M_PI * x) + 4.0 * M_PI * M_PI * sin(t) 
                                    * sin(M_PI * x) * cos(M_PI * x) + sin(t) * cos(M_PI * x) * cos(M_PI * y) 
                                    - 2.0 * sin(M_PI * x) * sin(M_PI * y) * sin(M_PI * y) * cos(t) * cos(M_PI * x));
        }

        virtual double
        value(const Point<dim> &p,
              const unsigned int component = 0) const override
        {
            double x = p[0];
            double y = p[1];
            double t = this->get_time();
            if (component == 0)
                return M_PI * (4.0 * M_PI * M_PI * sin(t) * sin(t) * sin(M_PI * x) * sin(M_PI * x) 
                                   * sin(M_PI * x) * sin(M_PI * y) * cos(M_PI * x) + 16.0 * M_PI 
                                   * M_PI * sin(t) * sin(M_PI * x) * sin(M_PI * x) * cos(M_PI * y) 
                                   - sin(t) * sin(M_PI * x) - 4.0 * M_PI * M_PI * sin(t) * cos(M_PI * y) 
                                   + 2.0 * sin(M_PI * x) * sin(M_PI * x) * cos(t) * cos(M_PI * y)) * sin(M_PI * y);
            else if (component == 1)
                return M_PI * (4.0 * M_PI * M_PI * sin(t) * sin(t) * sin(M_PI * x) * sin(M_PI * x) 
                                   * sin(M_PI * y) * sin(M_PI * y) * sin(M_PI * y) * cos(M_PI * y) 
                                   - 16.0 * M_PI * M_PI * sin(t) * sin(M_PI * x) * sin(M_PI * y) 
                                   * sin(M_PI * y) * cos(M_PI * x) + 4.0 * M_PI * M_PI * sin(t) * sin(M_PI * x) 
                                   * cos(M_PI * x) + sin(t) * cos(M_PI * x) * cos(M_PI * y) - 2.0 * sin(M_PI * x) 
                                   * sin(M_PI * y) * sin(M_PI * y) * cos(t) * cos(M_PI * x));
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
                values[0] = M_PI * sin(t) * sin(2.0 * M_PI * p[1]) * sin(M_PI * p[0]) * sin(M_PI * p[0]);
                values[1] = -M_PI * sin(t) * sin(2.0 * M_PI * p[0]) * sin(M_PI * p[1]) * sin(M_PI * p[1]);
            }

            virtual double
            value(const Point<dim> &p, const unsigned int component = 0) const override
            {
                double t = this->get_time();
                if (component == 0)
                    return M_PI * sin(t) * sin(2.0 * M_PI * p[1]) * sin(M_PI * p[0]) * sin(M_PI * p[0]);

                else if (component == 1)
                    return -M_PI * sin(t) * sin(2.0 * M_PI * p[0]) * sin(M_PI * p[1]) * sin(M_PI * p[1]);
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
                    result[1] = 2.0 * M_PI * M_PI * sin(t) * cos(2.0 * M_PI * y) * sin(M_PI * x) * sin(M_PI * x);
                }
                else if (component == 1)
                {
                    result[0] = -2.0 * M_PI * M_PI * sin(t) * cos(2.0 * M_PI * x) * sin(M_PI * y) * sin(M_PI * y);
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

            const double nu;
            mutable EthierSteinmanVelocity exact_velocity_n;
            mutable EthierSteinmanPressure exact_pressure_n;
            
        };

        mutable EthierSteinmanVelocity exact_velocity;
        mutable EthierSteinmanPressure exact_pressure;
        mutable EthierSteinmanNeumann neumann_condition;
    };

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

    // Number of MPI processes.
    const unsigned int mpi_size;

    // This MPI process.
    const unsigned int mpi_rank;

    ConditionalOStream pcout; // Parallel output stream.

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

    // extractors for velocity and pressure.
    FEValuesExtractors::Vector velocity;

    FEValuesExtractors::Scalar pressure;

    // diffusion coefficient
    const double nu = 1.0;

    // Mesh.
    parallel::fullydistributed::Triangulation<dim> mesh;

    // ExactSolution2D exact_solution;

    EthierSteinman exact_solution3D;

    // forcing term 2D
    ForcingTerm2D forcing_term;

    // Affine constraints.
    AffineConstraints<double> constraints;

    // Current time.
    double time;

    double lift;

    double drag;

    double l2_H1_error = 0.0;


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

    std::vector<IndexSet> block_owned_dofs; 

    // DoFs relevant to the current process (including ghost DoFs).
    IndexSet locally_relevant_dofs;

    std::vector<IndexSet> block_relevant_dofs; 

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

    // Stiffness matrix K.
    TrilinosWrappers::BlockSparseMatrix stiffness_matrix;

    TrilinosWrappers::BlockSparseMatrix pressure_mass;

    TrilinosWrappers::BlockSparseMatrix rhs_matrix;

    // Right-hand side vector in the linear system.
    TrilinosWrappers::MPI::BlockVector system_rhs;

    // System solution (without ghost elements).
    TrilinosWrappers::MPI::BlockVector solution_owned;

    // System solution (including ghost elements).
    TrilinosWrappers::MPI::BlockVector solution;


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
    , T(T_) 
    , deltat(deltat_)
    , velocity(0)
    , pressure(dim)
    , mesh(MPI_COMM_WORLD)
    , exact_solution3D(nu)
{
}

#endif