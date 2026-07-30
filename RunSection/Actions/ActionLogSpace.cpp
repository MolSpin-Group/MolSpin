/////////////////////////////////////////////////////////////////////////
// ActionLogSpace implementation (RunSection module)
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2025 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#include "ActionLogSpace.h"
#include "ObjectParser.h"

bool RunSection::ActionLogSpace::CalculatePoints(int n, double start, double stop)
{
    if (n < 1 || !std::isfinite(start) || !std::isfinite(stop))
        return false;

    m_Points = arma::logspace<arma::rowvec>(start, stop, n);
    return m_Points.n_elem == static_cast<arma::uword>(n) && m_Points.is_finite();
}

bool RunSection::ActionLogSpace::DoStep()
{
    // Make sure we have an ActionScaler to act on

    if (actionScaler == nullptr || !this->IsValid())
    {
        return false;
    }

    double val = 0;
    if (!GetPoint(val))
        return true; // All requested grid points have already been emitted.

    if (!this->actionScaler->Set(val))
    {
        return false;
    }
    return true;
}

bool RunSection::ActionLogSpace::DoValidate()
{
    std::string str;
    if (!this->Properties()->Get("actionscalar", str) && !this->Properties()->Get("scalar", str))
    {
        std::cout << "ERROR: No ActionScalar specified for the LogSpace action \"" << this->Name() << "\"!" << std::endl;
        return false;
    }
    int NumPoints = 0;
    if (!this->Properties()->Get("points", NumPoints))
    {
        std::cout << "ERROR: No Number of points specified for the LogSpace action \"" << this->Name() << "\"!" << std::endl;
        return false;
    }
    if (NumPoints < 1)
    {
        std::cout << "ERROR: LogSpace action \"" << this->Name() << "\" requires at least one point." << std::endl;
        return false;
    }
    double lowbounds = 0;
    if (!this->Properties()->Get("minvalue", lowbounds) && !this->Properties()->Get("min", lowbounds))
    {
        std::cout << "ERROR: No miniumum value specified for the LogSpace action \"" << this->Name() << "\"!" << std::endl;
        return false;
    }
    double upbounds = 0;
    if (!this->Properties()->Get("maxvalue", upbounds) && !this->Properties()->Get("max", upbounds))
    {
        std::cout << "ERROR: No maximum value specified for the LogSpace action \"" << this->Name() << "\"!" << std::endl;
        return false;
    }

    if (!std::isfinite(lowbounds) || !std::isfinite(upbounds) || lowbounds >= upbounds)
    {
        std::cout << "ERROR: Incorrect bounds specified for the LogSpace action \"" << this->Name() << "\"!" << std::endl;
        return false;
    }

    m_Num = NumPoints;
    m_Bounds = {lowbounds, upbounds};

    // Attemp to set the ActionVector
    if (!this->Scalar(str, &(this->actionScaler)))
    {
        std::cout << "ERROR: Could not find ActionScaler \"" << str << "\" specified for the LogSpace action \"" << this->Name() << "\"!" << std::endl;
        return false;
    }

    if (this->actionScaler->IsReadonly())
    {
        std::cout << "ERROR: Read only ActionScaler \"" << str << "\" specified for the LogSpace action \"" << this->Name() << "\"! Cannot act on this scaler!" << std::endl;
        return false;
    }

    if (!CalculatePoints(m_Num, m_Bounds.first, m_Bounds.second))
    {
        std::cout << "ERROR: Failed to construct finite points for the LogSpace action \"" << this->Name() << "\"!" << std::endl;
        return false;
    }

    m_Step = 0;
    // Actions execute between task runs. Install point zero now only when
    // calculation step 1 belongs to this grid; delayed grids retain the
    // user's initial value until their configured first step.
    if (this->first == 1)
    {
        double val = 0;
        if (!GetPoint(val) || !this->actionScaler->Set(val))
            return false;
    }
    return true;
}

bool RunSection::ActionLogSpace::Reset()
{
    m_Step = 0;
    double val = 0;
    return GetPoint(val) && this->actionScaler->Set(val);
}

RunSection::ActionLogSpace::ActionLogSpace(const MSDParser::ObjectParser &_parser, const std::map<std::string, ActionScalar> &_scaler, const std::map<std::string, ActionVector> &_vector)
    : Action(_parser, _scaler, _vector), actionScaler(nullptr)
{
    m_Step = 0;
    m_Num = 0;
    m_Bounds = {0.0, 0.0};
    m_Points.reset();
}

bool RunSection::ActionLogSpace::GetPoint(double &val)
{
    if (m_Step < 0 || m_Step >= m_Num || static_cast<arma::uword>(m_Step) >= m_Points.n_elem)
        return false;

    val = m_Points[m_Step];
    m_Step++;
    return true;
}
