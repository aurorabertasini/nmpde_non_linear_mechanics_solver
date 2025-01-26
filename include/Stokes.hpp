#ifndef STOKES_HPP
#define STOKES_HPP

#include "Linardo.hpp"

using namespace dealii;

// Class implementing a solver for the Stokes problem.
class Stokes : public Linardo
{
public:
  Stokes(const std::string  &mesh_file_name_,
         const unsigned int &degree_velocity_,
         const unsigned int &degree_pressure_,
         const double reynolds_number_)
    : Linardo(mesh_file_name_, reynolds_number_) 
    , degree_velocity(degree_velocity_)
    , degree_pressure(degree_pressure_)
    , inlet_velocity(H)
  {}

  // Function for the forcing term.
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
      values[0] = 4.0 * uM * p[1] * (H - p[1])/ (H * H);

      for (unsigned int i = 1; i < dim + 1; ++i)
        values[i] = 0.0;
    }

    virtual double
    value(const Point<dim> &p, const unsigned int component = 0) const override
    {
      if (component == 0)
        return 4.0 * uM * p[1] * (H - p[1])/ (H * H);
      else
        return 0.0;
    }

  protected:
    const double uM = 0.3;
    double H;
  };

  void setup() override;
  void assemble() override;
  void solve() override;
  void output() override;
  std::string get_output_directory();
  
  TrilinosWrappers::MPI::BlockVector get_solution() const
  {
    return solution;
  }


protected:
  

  // Outlet pressure [Pa].
  const double p_out = 0;
  
  // Height of the channel.
  const double H = 0.41;

  // Forcing term.
  ForcingTerm forcing_term;

  // Polynomial degree used for velocity.
  const unsigned int degree_velocity;

  // Polynomial degree used for pressure.
  const unsigned int degree_pressure;

  // Inlet velocity.
  InletVelocity inlet_velocity;
  
  // System matrix.
  TrilinosWrappers::BlockSparseMatrix system_matrix;

  // Pressure mass matrix, needed for preconditioning. We use a block matrix for
  // convenience, but in practice we only look at the pressure-pressure block.
  TrilinosWrappers::BlockSparseMatrix pressure_mass;

  // Right-hand side vector in the linear system.
  TrilinosWrappers::MPI::BlockVector system_rhs;

  // System solution (without ghost elements).
  TrilinosWrappers::MPI::BlockVector solution_owned;

  // System solution (including ghost elements).
  TrilinosWrappers::MPI::BlockVector solution;
};

#endif