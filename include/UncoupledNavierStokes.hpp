#ifndef UNCOUPLED_NAVIER_STOKES_HPP
#define UNCOUPLED_NAVIER_STOKES_HPP

#include "includes_file.hpp"

using namespace dealii;

template <unsigned int dim>
class UncoupledNavierStokes
{
public:
    UncoupledNavierStokes(
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
        inlet_velocity(H),
        computing_timer(MPI_COMM_WORLD, pcout,
                        TimerOutput::summary,
                        TimerOutput::wall_times)
    {
        this->nu = (2. / 3.) * inlet_velocity.get_u_max() * cylinder_radius / reynolds_number;
    }

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
            : Function<dim>(dim)
        {
            this->H = H;
            if constexpr (dim == 2)
                this->uM = 1.5;
            else
                this->uM = 2.25;
        }

        virtual void
        vector_value(const Point<dim> &p, Vector<double> &values) const override
        {
            if constexpr (dim == 2)
                values[0] = 4.0 * uM * p[1] * (H - p[1]) / (H * H);
            else
                values[0] = 16.0 * uM * p[1] * (H - p[1]) * p[2] * (H - p[2]) / (H * H * H * H);
            for (unsigned int i = 1; i < dim; ++i)
                values[i] = 0.0;
        }

        virtual double
        value(const Point<dim> &p, const unsigned int component = 0) const override
        {
            if (component == 0)
                if constexpr (dim == 2)
                    return 4.0 * uM * p[1] * (H - p[1]) / (H * H);
                else
                    return 16.0 * uM * p[1] * (H - p[1]) * p[2] * (H - p[2]) / (H * H * H * H);
            else
                return 0.0;
        }

        double get_u_max() const
        {
            return uM;
        }

    protected:
        double uM;
        double H;
    };

    class Function_Neumann : public Function<dim>
    {
    public:
        Function_Neumann()
            : Function<dim>(dim + 1)
        {
        }

        virtual void
        vector_value(const Point<dim> & /*p*/, Vector<double> &values) const override
        {
            values[0] = 0.0;

            for (unsigned int i = 1; i < dim + 1; ++i)
                values[i] = 0.0;
        }

        virtual double
        value(const Point<dim> & /*p*/, const unsigned int component = 0) const override
        {
            if (component == 0)
                return 0.0;
            else
                return 0.0;
        }
    };

    class Function_u0 : public Function<dim>
    {
    public:
        virtual void
        vector_value(const Point<dim> & /*p*/, Vector<double> &values) const override
        {
            values[0] = 0.0;

            for (unsigned int i = 1; i < dim + 1; ++i)
                values[i] = 0.0;
        }

        virtual double
        value(const Point<dim> & /*p*/, const unsigned int component = 0) const override
        {
            if (component == 0)
                return 0.0;
            else
                return 0.0;
        }
    };

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

    void pressure_update(bool rotational);

private:

    // Height of the channel.
    const double H = 0.41;

    Triangulation<dim> triangulation;

    // Velocity FE: Q2 vector
    FESystem<dim> fe_velocity;
    DoFHandler<dim> dof_handler_velocity;

    // Pressure FE: Q1 scalar
    FE_SimplexP<dim> fe_pressure;
    DoFHandler<dim> dof_handler_pressure;

    // Mesh file name.
    const std::string mesh_file_name;

    // Polynomial degree Velocity.
    const unsigned int degree_velocity;

    // Polynomial degree Pressure.
    const unsigned int degree_pressure;

    const double reynolds_number;

    // Final time.
    const double T;
    unsigned int timestep_number;
    double deltat;

    // Number of MPI processes.
    const unsigned int mpi_size;

    // This MPI process.
    const unsigned int mpi_rank;
    
    ConditionalOStream pcout;

    parallel::fullydistributed::Triangulation<dim> mesh;

    InletVelocity inlet_velocity;

    TimerOutput computing_timer;

    AffineConstraints<double> constraints_velocity;

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
    TrilinosWrappers::MPI::Vector u_star_divergence;
    TrilinosWrappers::MPI::Vector velocity_solution;
    TrilinosWrappers::MPI::Vector update_velocity_solution;
    TrilinosWrappers::MPI::Vector velocity_system_rhs;
    TrilinosWrappers::MPI::Vector velocity_update_rhs;

    TrilinosWrappers::MPI::Vector old_pressure;
    TrilinosWrappers::MPI::Vector deltap;
    TrilinosWrappers::MPI::Vector pressure_solution;
    TrilinosWrappers::MPI::Vector pressure_system_rhs;

    double time;

    // Viscosity
    double nu;

    unsigned int time_step = 0;


    const double cylinder_radius = 0.1;


    bool rotational = false; 


    std::vector<double> vec_drag;

    std::vector<double> vec_lift;

    std::vector<double> vec_drag_coeff;

    std::vector<double> vec_lift_coeff;

    double lift;

    double drag;

    const double rho = 1.0;
};

#endif

