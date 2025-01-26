#ifndef CONFIG_READER_HPP
#define CONFIG_READER_HPP

#include <string>
#include <regex>
#include <unordered_set>
#include <filesystem>


class ConfigReader
{
private:
    std::filesystem::path mesh2DPath; ///< Path to the 2D mesh
    std::filesystem::path mesh3DPath; ///< Path to the 3D mesh
    int degreeVelocity = 0;          ///< Degree of polynomial for velocity
    int degreePressure = 0;          ///< Degree of polynomial for pressure
    double simulationPeriod = 0.0;  ///< Simulation period (T)
    double timeStep = 0.0;           ///< Time step (deltat)
    double Re = 0.0;                 ///< Reynolds number
    std::regex pattern{};           ///< Regular expression for parsing config
    std::unordered_set<std::string> requiredVariables = {}; ///< Required variables

    [[nodiscard]] auto checkVariableExists(const std::string &variableName) const -> bool;
    void ensureVariablesSet();
    void promptUserForVariable(const std::string& variableName);


public:
    ConfigReader();

    [[nodiscard]] auto getMesh2DPath() const -> std::filesystem::path;
    [[nodiscard]] auto getMesh3DPath() const -> std::filesystem::path;
    [[nodiscard]] auto getDegreeVelocity() const -> int;
    [[nodiscard]] auto getDegreePressure() const -> int;
    [[nodiscard]] auto getSimulationPeriod() const -> double;
    [[nodiscard]] auto getTimeStep() const -> double;
    [[nodiscard]] auto readConfigFile() -> bool;
    [[nodiscard]] auto getRe() const -> double;
    };

#endif 