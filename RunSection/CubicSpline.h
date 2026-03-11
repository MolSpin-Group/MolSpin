/////////////////////////////////////////////////////////////////////////
// Utility-CubicSpline (RunSection module)
// ------------------
// template header file for the cubic spline used in interpolation
// 
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2025 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#ifndef MOD_RunSection_Utility_CS
#define MOD_RunSection_Utility_CS

#include "SpinAPIfwd.h"
#include "SpinSpace.h"
#include <utility>

namespace RunSection
{
    enum class ValType
    {
        scaler = 0,
        vector = 1,
        matrix = 2
    };
    template<typename t>
        class Buffer 
        {
        public:
            ValType type;
            std::vector<t> rhoPoints;
        private:
            int max;
            std::vector<double> timePoints;
        public:
            Buffer(int n, ValType ty)
                :max(n), type(ty)
            {}
            void push(double time, t element) {
                timePoints.push_back(time);
                rhoPoints.push_back(t);
                if((int)timePoints.size() > max)
                {
                    timePoints.erase(timePoints.begin());
                    rhoPoints.erase(rhoPoints.begin());
                }
            }
            int size() const {return (int)timePoints.size()}
            arma::vec time() {
                arma::vec tvec(size());
                for(int i = 0; i < size(); i++)
                {
                    tvec(i) = timePoints[i];
                }
                return tvec;
            }
        };

        class ScalerSpline
        {
        public:
            arma::vec timePoints;
            arma::cx_vec y;
            arma::cx_vec m;

            void Build(const arma::vec& t, const arma::cx_vec& yPoints);
            arma::cx_double Eval(double T) const;
        private:
            arma::cx_double EvalFull(double T) const;
            arma::cx_double Eval2points(double T) const;
        };

        class MatrixSpline
        {
        public:
            std::vector<ScalerSpline> splines;
            int dim;

            void build(const arma::cx_vec t, const std::vector<arma::cx_mat>& rho_hist);
            arma::cx_mat Eval(double T) const;
        }

}
#endif