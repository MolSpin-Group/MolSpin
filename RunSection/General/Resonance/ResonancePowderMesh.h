/////////////////////////////////////////////////////////////////////////
// Conservative resonance-surface powder integration.
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2026 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#ifndef MOLSPIN_RESONANCE_POWDER_MESH_H
#define MOLSPIN_RESONANCE_POWDER_MESH_H

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace RunSection::General::Resonance
{
    // Vertex data from exact diagonalization. All ordered level pairs are kept,
    // including zero-intensity ones; pair indices must agree between vertices.
    struct ResonanceMeshNode
    {
        std::vector<double> omega;
        std::vector<std::vector<double>> strength; // population difference * moment
    };

    // Integrate a piecewise-linear frequency resonance surface in
    // (cos(theta), phi, B). A cell is six tetrahedra. No inverse field slope is
    // sampled: the delta-function Jacobian is the gradient in all three mesh
    // coordinates. The intersection polygon is projected conservatively into
    // field bins; Gaussian broadening is applied separately by the caller.
    class ResonancePowderMesh
    {
        using V=std::array<double,3>;
        struct Point { V x; double b; std::vector<double> w; };
        double first, step;
        std::vector<std::vector<double>> &mass;
        static V Sub(V a,V b) { return {a[0]-b[0],a[1]-b[1],a[2]-b[2]}; }
        static V Cross(V a,V b) { return {a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]}; }
        static double Dot(V a,V b) { return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; }
        static double Norm(V a) { return std::sqrt(Dot(a,a)); }

        void DepositTriangle(Point a,Point b,Point c,double measure)
        {
            std::array<Point,3> p={std::move(a),std::move(b),std::move(c)};
            std::sort(p.begin(),p.end(),[](const auto &a,const auto &b){return a.b<b.b;});
            const double low=p[0].b,mid=p[1].b,high=p[2].b;
            if (high-low<1e-13)
            {
                const double index=(low-first)/step;
                const int left=static_cast<int>(std::floor(index));
                const double f=index-left;
                for (size_t k=0;k<mass.size();++k)
                {
                    const double w=measure*(p[0].w[k]+p[1].w[k]+p[2].w[k])/3.;
                    if (left>=0 && left<static_cast<int>(mass[k].size())) mass[k][left]+=(1-f)*w;
                    if (left+1>=0 && left+1<static_cast<int>(mass[k].size())) mass[k][left+1]+=f*w;
                }
                return;
            }
            const int firstBin=std::max(0,static_cast<int>(std::floor((low-first)/step+.5)));
            const int lastBin=std::min(static_cast<int>(mass[0].size())-1,static_cast<int>(std::floor((high-first)/step+.5)));
            for (int j=firstBin;j<=lastBin;++j)
            {
                const double l=std::max(low,first+(j-.5)*step),r=std::min(high,first+(j+.5)*step);
                if (!(r>l)) continue;
                for (size_t k=0;k<mass.size();++k)
                {
                    double value=0.;
                    // Integral of the triangle's linearly weighted B density.
                    if (mid>low && l<mid)
                    {
                        const double x0=l-low,x1=std::min(r,mid)-low;
                        const double den=(mid-low)*(high-low);
                        const double slope=(p[1].w[k]-p[0].w[k])/(mid-low)+(p[2].w[k]-p[0].w[k])/(high-low);
                        value+=(p[0].w[k]*(x1*x1-x0*x0)+slope*(x1*x1*x1-x0*x0*x0)/3.)/den;
                    }
                    if (high>mid && r>mid)
                    {
                        const double x0=high-r,x1=high-std::max(l,mid);
                        const double den=(high-mid)*(high-low);
                        const double slope=(p[1].w[k]-p[2].w[k])/(high-mid)+(p[0].w[k]-p[2].w[k])/(high-low);
                        value+=(p[2].w[k]*(x1*x1-x0*x0)+slope*(x1*x1*x1-x0*x0*x0)/3.)/den;
                    }
                    mass[k][j]+=measure*value;
                }
            }
        }

        void DepositTetrahedron(const std::array<const ResonanceMeshNode*,4> &nodes,
            const std::array<double,4> &fields,size_t pair,double omega,double determinant)
        {
            std::array<double,4> f;
            for (size_t i=0;i<4;++i) f[i]=nodes[i]->omega[pair]-omega;
            const auto limits=std::minmax_element(f.begin(),f.end());
            if (*limits.first>0 || *limits.second<=0) return;
            const V gradient={f[1]-f[0],f[2]-f[0],f[3]-f[0]};
            const double length=Norm(gradient);
            if (!(length>0)) return;
            static const std::array<V,4> vertices={V{0,0,0},V{1,0,0},V{0,1,0},V{0,0,1}};
            std::vector<Point> polygon;
            for (size_t i=0;i<4;++i) for (size_t j=i+1;j<4;++j)
            {
                if ((f[i]>0)==(f[j]>0)) continue;
                const double t=f[i]/(f[i]-f[j]);
                Point p;
                for (size_t k=0;k<3;++k) p.x[k]=(1-t)*vertices[i][k]+t*vertices[j][k];
                bool duplicate=false;
                for (const auto &q:polygon) if (Norm(Sub(p.x,q.x))<1e-12) duplicate=true;
                if (duplicate) continue;
                p.b=(1-t)*fields[i]+t*fields[j];
                p.w.resize(mass.size());
                for (size_t k=0;k<mass.size();++k) p.w[k]=(1-t)*nodes[i]->strength[pair][k]+t*nodes[j]->strength[pair][k];
                polygon.push_back(std::move(p));
            }
            if (polygon.size()<3) return;
            if (polygon.size()==4)
            {
                V center={0,0,0};
                for (const auto &p:polygon) for(size_t k=0;k<3;++k) center[k]+=p.x[k]/4.;
                const V u=Sub(polygon[0].x,center),v=Cross(gradient,u);
                std::sort(polygon.begin(),polygon.end(),[&](const auto &a,const auto &b) {
                    return std::atan2(Dot(Sub(a.x,center),v)/length,Dot(Sub(a.x,center),u))
                         < std::atan2(Dot(Sub(b.x,center),v)/length,Dot(Sub(b.x,center),u));
                });
            }
            for (size_t j=1;j+1<polygon.size();++j)
            {
                const double area=.5*Norm(Cross(Sub(polygon[j].x,polygon[0].x),Sub(polygon[j+1].x,polygon[0].x)));
                DepositTriangle(polygon[0],polygon[j],polygon[j+1],determinant*area/length);
            }
        }

    public:
        // Field-bin centers in tesla; mass units are strength * T/(rad/ns).
        ResonancePowderMesh(double firstField,double fieldStep,std::vector<std::vector<double>> &binMass)
            :first(firstField),step(fieldStep),mass(binMass) {}

        void AddCell(const std::array<const ResonanceMeshNode*,8> &nodes,
            double lowField,double highField,double du,double dphi,double omega)
        {
            static const int tetra[6][4]={{0,1,3,7},{0,1,5,7},{0,2,3,7},{0,2,6,7},{0,4,5,7},{0,4,6,7}};
            const double determinant=du*dphi*(highField-lowField);
            for (size_t pair=0;pair<nodes[0]->omega.size();++pair)
            {
                double lo=nodes[0]->omega[pair],hi=lo;
                for(size_t v=1;v<8;++v) {lo=std::min(lo,nodes[v]->omega[pair]);hi=std::max(hi,nodes[v]->omega[pair]);}
                if (lo>omega || hi<=omega) continue;
                for (const auto &t:tetra)
                {
                    std::array<const ResonanceMeshNode*,4> n;
                    std::array<double,4> b;
                    for(size_t k=0;k<4;++k) {n[k]=nodes[t[k]];b[k]=(t[k]&4)?highField:lowField;}
                    DepositTetrahedron(n,b,pair,omega,determinant);
                }
            }
        }
    };
}
#endif
