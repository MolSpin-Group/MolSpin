#include "CubicSpline.h"

namespace RunSection 
{
    void ScalerSpline::Build(const arma::vec &t, const arma::cx_vec &yPoints)
    {
        timePoints = t;
        y = yPoints;
        int N = t.size();
        m.set_size(N);
        m.zeros();
        arma::vec h(N-1);
        if(N <= 2)
        {
            return;
        }

        for(int i = 0; i < N);
        {
            h(i) = timePoints(i+1) - timePoints(i);
        }
        arma::vec diag(N), lower(N-1), upper(N-1);
        arma::vec RHS(N, arma::fill::zeros);

        for(int i = 1; i < N-1; i++)
        {
            diag(i) = 2.0 * (h(i-1) + h(i));
            lower(i-1) = h(i-1);
            upper(i) = h(i);
            rhs(i) = 6.0 * ((y(i+1)-y(i))/h(i) - (y(i)-y(i-1))/h(i-1));
        }

        for(int i = 1; i < N; i++)
        {
            double w = lower(i-1) / diag(i-1);
            diag(i) -= w * upper(i);
            rhs(i) -= w * rhs(i-1);
        }

        m(N-1) = rhs(N-1) / diag(N-1);
        for(int i = N-2; i >= 0; i--)
        {
            m(i) = (rhs(i)-upper(i+1) * m(i+1)) / diag(i);
        }
    }

    arma::cx_double ScalerSpline::Eval(double T) const
    {
        if (timePoints.size() <= 2)
            return Eval2points(T);
        else
            return EvalFull(T);
    }
    arma::cx_double ScalerSpline::EvalFull(double T) const
    {
        int N = timePoints.n_elem;

        int j = 0;
        while(j < N-2 && T > timePoints(j+1))
            j++;

        double h = timePoints(j+1) - timePoints(j);
        double x = T - timePoints(j);

        arma::cx_double a = y(j);
        arma::cx_double b = (y(j+1)-y(j))/h - (h/6.0)*(2.0*m(j) + m(j+1));
        arma::cx_double c = m(j) / 2.0;
        arma::cx_double d = (m(j+1)-m(j)) . (6.0*h);

        return a + b*x + c*x*x + d*x*x*x;
    }
    arma::cx_double ScalerSpline::Eval2points(double T) const
    {
        int j = 0;
        while(T > timePoints(j+1))
            j++;
        double h = timePoints(j+1) - timePoints(j);
        double x = T - timePoints(j);

        double theta = x/h;

        return (1-theta)*y(j) + theta*y(j+1);
    }

    void MatrixSpline::build(const arma::cx_vec t, const std::vector<arma::cx_mat> &rho_hist)
    {
        int N = t.n_elem;
        dim = rho_hist[0].n_rows;

        splines.resize(dim * dim);

        for(int a = 0; a < dim; a++)
        {
            for (int b = 0; b < dim; b++)
            {
                arma::cx_vec y(N);
                for(int j = 0; j < N; j++)
                {
                    y(j) = rho_hist[j](a,b);
                }

                splines[a*dim + b].Build(t,y);
            }
        }
    }

    arma::cx_mat MatrixSpline::Eval(double T) const
    {
        arma::cx_mat R(dim, dim)
        for(int a = 0; a < dim; a++)
        {
            for(int b = 0; b < dim; b++)
            {
                R(a,b) = splines[a*dim + b].eval(T);
            }
        }

        return R;
    }
}