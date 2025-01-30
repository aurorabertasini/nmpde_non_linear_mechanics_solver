#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mpi.h>

#include <deal.II/base/mpi.h>
#include <deal.II/lac/affine_constraints.h> // instead of "constraint_matrix.h"
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>

#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>

#include <fstream>
#include <iostream>
#include <memory> // for std::shared_ptr
#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/conditional_ostream.h>

#include <deal.II/base/mpi.h>
#include <deal.II/lac/affine_constraints.h> // instead of "constraint_matrix.h"
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>

#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>

#include <fstream>
#include <iostream>
#include <memory> // for std::shared_ptrù

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

#include <filesystem>
#include <fstream>
#include <iostream>

using namespace dealii;

template <unsigned int dim>
class IncrementalChorinTemam
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
            else
                return factor * (std::exp(a * x3) * std::sin(a * x1 + b * x2) + std::exp(a * x2) * std::cos(a * x3 + b * x1));
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
        const double a = M_PI / 4.;
        const double b = M_PI / 2.;
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
        const double a = M_PI / 4.;
        const double b = M_PI / 2.;
        const double nu;
    };

    class EthierSteinmanNeumann : public Function<dim>
    {
    public:
        EthierSteinmanNeumann(double nu_)
            : Function<dim>(dim + 1), nu(nu_), exact_velocity(nu_), exact_pressure(nu_)
        {
        }

        virtual double
        value(const Point<dim> &p, const unsigned int component) const
        {
            exact_pressure.set_time(this->get_time());
            exact_velocity.set_time(this->get_time());

            if (component == 0 || component == 2)
            {
                Tensor<1, dim> velocity_gradient =
                    exact_velocity.gradient(p, component);
                return -nu * velocity_gradient[1];
            }
            else if (component == 1)
            {
                Tensor<1, dim> velocity_gradient =
                    exact_velocity.gradient(p, component);
                return -nu * velocity_gradient[1] + exact_pressure.value(p);
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

    private:
        mutable EthierSteinmanVelocity exact_velocity;
        mutable EthierSteinmanPressure exact_pressure;

        const double nu;
    };

    IncrementalChorinTemam(const std::string &mesh_file_name_,
                           const unsigned int &degree_velocity_,
                           const unsigned int &degree_pressure_,
                           const double &T_,
                           const double &deltat_,
                           const double &reynolds_number_);

    void setup();

    void assemble_system_velocity();

    void solve_velocity_system();

    void assemble_system_pressure();

    void solve_pressure_system();

    void update_velocity();

    void solve_update_velocity_system();

    void output_results();

    void run();

    void compute_lift_drag();

    std::string get_output_directory();

    double compute_error(const VectorTools::NormType &norm_type);

    void update_buondary_conditions();

private:
    Triangulation<dim> triangulation;

    // Velocity FE: Q2 vector
    FESystem<dim> fe_velocity;
    DoFHandler<dim> dof_handler_velocity;
    AffineConstraints<double> constraints_velocity;

    // Pressure FE: Q1 scalar
    FE_SimplexP<dim> fe_pressure;
    DoFHandler<dim> dof_handler_pressure;
    AffineConstraints<double> constraints_pressure;

    // Owned & relevant dofs
    IndexSet locally_owned_velocity;
    IndexSet locally_relevant_velocity;
    IndexSet locally_owned_pressure;
    IndexSet locally_relevant_pressure;

    // System matrices
    TrilinosWrappers::SparseMatrix velocity_matrix;
    TrilinosWrappers::SparseMatrix pressure_matrix;
    TrilinosWrappers::SparseMatrix velocity_update_matrix;

    // System vectors
    TrilinosWrappers::MPI::Vector old_velocity;
    TrilinosWrappers::MPI::Vector old_old_velocity;
    TrilinosWrappers::MPI::Vector u_star;
    TrilinosWrappers::MPI::Vector velocity_solution;
    TrilinosWrappers::MPI::Vector update_velocity_solution;
    TrilinosWrappers::MPI::Vector velocity_system_rhs;
    TrilinosWrappers::MPI::Vector velocity_update_rhs;

    TrilinosWrappers::MPI::Vector old_pressure;
    TrilinosWrappers::MPI::Vector deltap;
    TrilinosWrappers::MPI::Vector pressure_solution;
    TrilinosWrappers::MPI::Vector pressure_system_rhs;

    // Final time.
    const double T;
    unsigned int timestep_number;
    double deltat;
    double time = 0;

    // Viscosity
    const double nu = 1.;

    unsigned int time_step = 0;

    const double reynolds_number;

    const double cylinder_radius = 0.1;

    ConditionalOStream pcout;

    TimerOutput computing_timer;

    // Mesh file name.
    const std::string mesh_file_name;

    // Polynomial degree Velocity.
    const unsigned int degree_velocity;

    // Polynomial degree Pressure.
    const unsigned int degree_pressure;

    // Number of MPI processes.
    const unsigned int mpi_size;

    // This MPI process.
    const unsigned int mpi_rank;

    parallel::fullydistributed::Triangulation<dim> mesh;

    EthierSteinmanVelocity exact_velocity3D;

    EthierSteinmanPressure exact_pressure3D;

    EthierSteinmanNeumann neumann_function3D;

    ExactVelocity2D exact_velocity2D;

    ExactPressure2D exact_pressure2D;

    ForcingTerm2D forcing_term2D;

    // Height of the channel.
    const double H = 0.41;

    std::vector<double> vec_drag;

    std::vector<double> vec_lift;

    std::vector<double> vec_drag_coeff;

    std::vector<double> vec_lift_coeff;

    double lift;

    double drag;

    const double rho = 1.0;
};

template <unsigned int dim>
IncrementalChorinTemam<dim>::IncrementalChorinTemam(
    const std::string &mesh_file_name_,
    const unsigned int &degree_velocity_,
    const unsigned int &degree_pressure_,
    const double &T_,
    const double &deltat_,
    const double &reynolds_number_)
    : fe_velocity(FE_SimplexP<dim>(2), dim),
      dof_handler_velocity(triangulation),
      fe_pressure(1),
      dof_handler_pressure(triangulation),
      mesh_file_name(mesh_file_name_),
      degree_velocity(degree_velocity_),
      degree_pressure(degree_pressure_),
      reynolds_number(reynolds_number_),
      T(T_),
      deltat(deltat_),
      mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)),
      mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)),
      pcout(std::cout, mpi_rank == 0),
      mesh(MPI_COMM_WORLD),
      computing_timer(MPI_COMM_WORLD, pcout,
                      TimerOutput::summary,
                      TimerOutput::wall_times),
      exact_velocity3D(nu),
      exact_pressure3D(nu),
      neumann_function3D(nu)
{
}